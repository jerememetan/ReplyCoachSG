"""LoRA-only Qwen reranker training on the existing validated labelled pairs."""
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
from time import perf_counter

import torch
from replycoach.reranker_bundle import load_training_bundle, pair_payload
from replycoach.reranker_hard_pairs import apply_reviewed_pairs
from replycoach.qwen_reranker import MODEL, REVISION, PROMPT_SHA256, file_sha256, load_base, score_batch


def parameter_hash(model, *, trainable):
    digest=sha256()
    for name, value in sorted(model.named_parameters()):
        if value.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_training(bundle_path, output, *, source=MODEL, epochs=4, batch_size=2,
                 accumulation=8, learning_rate=2e-5, smoke=False, max_steps=None,
                 model_factory=None):
    from peft import LoraConfig, get_peft_model
    bundle_path, output=Path(bundle_path), Path(output)
    bundle=load_training_bundle(bundle_path)
    reviewed=json.loads((bundle_path/'reviewed_pairs.json').read_text()) if (bundle_path/'reviewed_pairs.json').is_file() else None
    train, review_meta=apply_reviewed_pairs(bundle,reviewed) if reviewed else (bundle['train'], {'method':'none'})
    if output.exists() or bundle_path.resolve() in output.resolve().parents:
        raise ValueError('output must be a fresh directory outside the input bundle')
    if (type(epochs) is not int or not 1 <= epochs <= 4 or type(batch_size) is not int
            or batch_size < 1 or type(accumulation) is not int or accumulation < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('invalid fixed Qwen training settings')
    if not smoke and (model_factory is not None or max_steps is not None or not torch.cuda.is_available()):
        raise ValueError('production Qwen training requires GPU and disallows smoke overrides')
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError('smoke step count must be positive')
    device='cpu' if smoke else 'cuda'
    torch.manual_seed(425)
    if model_factory:
        model, tokenizer=model_factory()
        base_file_hash='test-only'
    else:
        local=Path(source)/'model.safetensors'
        if not local.is_file():
            from huggingface_hub import hf_hub_download
            local=Path(hf_hub_download(MODEL,'model.safetensors',revision=REVISION))
        base_file_hash=file_sha256(local)
        model,tokenizer=load_base(source,device)
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,
                                        lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none'))
    if any('lora_' not in name for name,p in model.named_parameters() if p.requires_grad):
        raise ValueError('only LoRA adapter parameters may train')
    if not smoke:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    frozen_hash=parameter_hash(model,trainable=False)
    initial_hash=parameter_hash(model,trainable=True)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=learning_rate,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=not smoke)
    output.mkdir(parents=True)
    (output/'training_pairs.jsonl').write_bytes(pair_payload(train))
    validation=bundle['validation'][:4] if smoke else bundle['validation']
    steps,history,best_loss,best_epoch,best_hash=0,[],float('inf'),0,None
    started=perf_counter()
    for epoch in range(1,epochs+1):
        order=list(train)
        random.Random(425+epoch).shuffle(order)
        model.train()
        total,count=0.,0
        for offset in range(0,len(order),batch_size*accumulation):
            window=order[offset:offset+batch_size*accumulation]
            optimizer.zero_grad(set_to_none=True)
            for start in range(0,len(window),batch_size):
                rows=window[start:start+batch_size]
                scores,_=score_batch(model,tokenizer,[(p['query'],p['reply']) for p in rows])
                labels=torch.tensor([p['label'] for p in rows],device=device,dtype=torch.float32)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(scores,labels)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite Qwen training loss')
                scaler.scale(loss*len(rows)/len(window)).backward()
                total+=float(loss.detach())*len(rows)
                count+=len(rows)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            steps+=1
            if steps % 20 == 0:
                print(json.dumps({'event':'qwen_step','epoch':epoch,'steps':steps}),flush=True)
            if max_steps and steps >= max_steps:
                break
        model.eval()
        validation_loss=0.
        with torch.inference_mode():
            for start in range(0,len(validation),batch_size):
                rows=validation[start:start+batch_size]
                scores,_=score_batch(model,tokenizer,[(p['query'],p['reply']) for p in rows])
                loss=torch.nn.functional.binary_cross_entropy_with_logits(
                    scores,torch.tensor([p['label'] for p in rows],device=device,dtype=torch.float32))
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite Qwen validation loss')
                validation_loss+=float(loss)*len(rows)
        validation_loss/=len(validation)
        history.append({'epoch':epoch,'train_loss':total/count,'validation_loss':validation_loss,
                        'optimizer_steps':steps,'elapsed_seconds':perf_counter()-started})
        print(json.dumps({'event':'qwen_epoch',**history[-1]}),flush=True)
        if validation_loss < best_loss:
            best_loss,best_epoch=validation_loss,epoch
            best=output/'best'
            model.save_pretrained(best,safe_serialization=True)
            tokenizer.save_pretrained(best)
            best_hash=parameter_hash(model,trainable=True)
        if max_steps and steps >= max_steps:
            break
    base_unchanged=parameter_hash(model,trainable=False)==frozen_hash
    if not base_unchanged or best_hash==initial_hash or not steps:
        raise ValueError('Qwen training must update adapters and preserve all frozen weights')
    result={'record_type':'qwen_lora_reranker_training','completed':True,
            'production_checkpoint':not smoke,'base_model':MODEL,'base_revision':REVISION,
            'base_weights_sha256':base_file_hash,'prompt_sha256':PROMPT_SHA256,
            'device':device,'device_name':torch.cuda.get_device_name(0) if not smoke else 'CPU smoke only',
            'train_fingerprint':bundle['manifest']['train_fingerprint'],
            'validation_fingerprint':bundle['manifest']['validation_fingerprint'],
            'training_pairs_sha256':sha256(pair_payload(train)).hexdigest(),
            'validation_pairs_sha256':bundle['manifest']['pair_sha256']['validation'],
            'train_pair_count':len(train),'validation_pair_count':len(validation),
            'reviewed_pairs':review_meta,'optimizer_steps':steps,'best_epoch':best_epoch,
            'best_validation_loss':best_loss,'history':history,'seed':425,'epochs':epochs,
            'batch_size':batch_size,'accumulation':accumulation,'learning_rate':learning_rate,
            'weight_decay':.01,'max_length':512,'lora_rank':8,'lora_alpha':16,'lora_dropout':.05,
            'trainable_parameters':sum(p.numel() for p in trainable),'base_unchanged':base_unchanged,
            'initial_adapter_parameters_sha256':initial_hash,'best_adapter_parameters_sha256':best_hash,
            'weights_changed':True,'adapter_sha256':file_sha256(output/'best/adapter_model.safetensors'),
            'elapsed_seconds':perf_counter()-started,
            'packages':{name:version(name) for name in ['torch','transformers','peft','tokenizers','safetensors']}}
    (output/'best/qwen_training.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
