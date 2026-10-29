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

MAX_OVERFLOW_RETRIES = 8


def finish_optimizer_step(optimizer, scaler, trainable):
    """Never clip or apply invalid gradients; let AMP back off before a retry."""
    scaler.unscale_(optimizer)
    gradients=[p.grad for p in trainable if p.grad is not None]
    if not gradients:
        raise RuntimeError('Qwen optimizer has no gradients')
    finite=bool(torch.stack([torch.isfinite(grad).all() for grad in gradients]).all())
    before=float(scaler.get_scale())
    if not finite:
        if not scaler.is_enabled():
            raise RuntimeError('nonfinite Qwen gradients without active loss scaling')
        # unscale_ recorded the inf/NaN. GradScaler must skip this optimizer step,
        # then update its scale. Do not call optimizer.step or clip these gradients.
        scaler.step(optimizer)
        scaler.update()
        after=float(scaler.get_scale())
        if not math.isfinite(after) or not 0 < after < before:
            raise RuntimeError('Qwen gradient overflow did not reduce the loss scale')
        return {'updated':False,'loss_scale_before':before,'loss_scale_after':after}
    torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return {'updated':True,'loss_scale_before':before,'loss_scale_after':float(scaler.get_scale())}


def parameter_hash(model, *, trainable):
    digest=sha256()
    for name, value in sorted(model.named_parameters()):
        if value.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_training(bundle_path, output, *, source=MODEL, epochs=4, batch_size=2,
                 accumulation=8, learning_rate=2e-5, smoke=False, max_steps=None,
                 model_factory=None, early_stopping_patience=0, initial_adapter=None):
    bundle_path, output=Path(bundle_path), Path(output)
    bundle=load_training_bundle(bundle_path)
    reviewed=json.loads((bundle_path/'reviewed_pairs.json').read_text()) if (bundle_path/'reviewed_pairs.json').is_file() else None
    train, review_meta=apply_reviewed_pairs(bundle,reviewed) if reviewed else (bundle['train'], {'method':'none'})
    return fit_pairs(train,bundle['validation'],bundle['manifest'],output,input_path=bundle_path,
                     source=source,epochs=epochs,batch_size=batch_size,accumulation=accumulation,
                     learning_rate=learning_rate,smoke=smoke,max_steps=max_steps,
                     model_factory=model_factory,early_stopping_patience=early_stopping_patience,
                     initial_adapter=initial_adapter,review_meta=review_meta)


def fit_pairs(train, validation, manifest, output, *, input_path, source=MODEL,
              epochs=4, batch_size=2, accumulation=8, learning_rate=2e-5,
              smoke=False, max_steps=None, model_factory=None, early_stopping_patience=0,
              initial_adapter=None, review_meta=None):
    """Shared optimizer; callers validate their project or external-source schema first."""
    from peft import LoraConfig, get_peft_model, PeftModel
    output, input_path = Path(output), Path(input_path)
    if (output.exists() or input_path.resolve() in output.resolve().parents
            or output.resolve() in input_path.resolve().parents):
        raise ValueError('output must be a fresh directory outside the input bundle')
    if not train or not validation:
        raise ValueError('training and validation pairs must be nonempty')
    if (type(epochs) is not int or not 1 <= epochs <= 12 or type(batch_size) is not int
            or batch_size < 1 or type(accumulation) is not int or accumulation < 1
            or type(early_stopping_patience) is not int or not 0 <= early_stopping_patience <= 12
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
    initialization={'method':'fresh_lora'}
    if initial_adapter is not None:
        adapter=Path(initial_adapter)
        meta=json.loads((adapter/'qwen_training.json').read_text(encoding='utf-8'))
        if (meta.get('completed') is not True or meta.get('production_checkpoint') != (not smoke)
                or meta.get('base_model') != MODEL or meta.get('base_revision') != REVISION
                or meta.get('prompt_sha256') != PROMPT_SHA256
                or meta.get('base_weights_sha256') != base_file_hash
                or meta.get('base_unchanged') is not True or meta.get('weights_changed') is not True
                or meta.get('adapter_sha256') != file_sha256(adapter/'adapter_model.safetensors')):
            raise ValueError('initial adapter provenance or weights do not match')
        model=PeftModel.from_pretrained(model,adapter,is_trainable=True)
        if parameter_hash(model,trainable=True) != meta['best_adapter_parameters_sha256']:
            raise ValueError('loaded adapter parameters do not match selected checkpoint')
        initialization={'method':'continue_selected_adapter','adapter_sha256':meta['adapter_sha256'],
                        'training_metadata_sha256':file_sha256(adapter/'qwen_training.json'),
                        'train_fingerprint':meta['train_fingerprint'],'best_epoch':meta['best_epoch']}
    else:
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
    validation=validation[:4] if smoke else validation
    steps,history,best_loss,best_epoch,best_hash=0,[],float('inf'),0,None
    overflow_retries=0
    stale_epochs,stop_reason=0,'epoch_budget'
    started=perf_counter()
    for epoch in range(1,epochs+1):
        order=list(train)
        random.Random(425+epoch).shuffle(order)
        model.train()
        total,count=0.,0
        for offset in range(0,len(order),batch_size*accumulation):
            window=order[offset:offset+batch_size*accumulation]
            for attempt in range(MAX_OVERFLOW_RETRIES+1):
                optimizer.zero_grad(set_to_none=True)
                window_loss=0.
                for start in range(0,len(window),batch_size):
                    rows=window[start:start+batch_size]
                    scores,_=score_batch(model,tokenizer,[(p['query'],p['reply']) for p in rows])
                    labels=torch.tensor([p['label'] for p in rows],device=device,dtype=torch.float32)
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(scores,labels)
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite Qwen training loss')
                    scaler.scale(loss*len(rows)/len(window)).backward()
                    window_loss+=float(loss.detach())*len(rows)
                update=finish_optimizer_step(optimizer,scaler,trainable)
                if update['updated']:
                    break
                overflow_retries+=1
                event={'event':'qwen_gradient_overflow','epoch':epoch,'steps':steps,
                       'window_start':offset,'attempt':attempt+1,
                       'will_retry':attempt<MAX_OVERFLOW_RETRIES,**update}
                print(json.dumps(event),flush=True)
                with (output/'overflow_events.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(event)+'\n')
                if attempt==MAX_OVERFLOW_RETRIES:
                    raise RuntimeError(f'Qwen gradient overflow persisted after {MAX_OVERFLOW_RETRIES} retries; '
                                       'inspect overflow_events.jsonl; no invalid update was applied')
            total+=window_loss
            count+=len(window)
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
                        'optimizer_steps':steps,'gradient_overflow_retries':overflow_retries,
                        'loss_scale':float(scaler.get_scale()),'elapsed_seconds':perf_counter()-started})
        print(json.dumps({'event':'qwen_epoch',**history[-1]}),flush=True)
        if validation_loss < best_loss:
            stale_epochs=0
            best_loss,best_epoch=validation_loss,epoch
            best=output/'best'
            model.save_pretrained(best,safe_serialization=True)
            tokenizer.save_pretrained(best)
            best_hash=parameter_hash(model,trainable=True)
        else:
            stale_epochs+=1
        (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if max_steps and steps >= max_steps:
            stop_reason='smoke_step_limit'
            break
        if early_stopping_patience and stale_epochs >= early_stopping_patience:
            stop_reason='validation_patience'
            break
    base_unchanged=parameter_hash(model,trainable=False)==frozen_hash
    if not base_unchanged or best_hash==initial_hash or not steps:
        raise ValueError('Qwen training must update adapters and preserve all frozen weights')
    result={'record_type':'qwen_lora_reranker_training','completed':True,
            'production_checkpoint':not smoke,'base_model':MODEL,'base_revision':REVISION,
            'base_weights_sha256':base_file_hash,'prompt_sha256':PROMPT_SHA256,
            'device':device,'device_name':torch.cuda.get_device_name(0) if not smoke else 'CPU smoke only',
            'train_fingerprint':manifest['train_fingerprint'],
            'validation_fingerprint':manifest['validation_fingerprint'],
            'training_pairs_sha256':sha256(pair_payload(train)).hexdigest(),
            'validation_pairs_sha256':manifest['pair_sha256']['validation'],
            'train_pair_count':len(train),'validation_pair_count':len(validation),
            'reviewed_pairs':review_meta or {'method':'none'},'initialization':initialization,
            'source_manifest':manifest if manifest.get('record_type')=='dailydialog_reranker_adaptation' else None,
            'optimizer_steps':steps,'best_epoch':best_epoch,
            'gradient_policy':'finite_only_amp_retry_v1','gradient_overflow_retries':overflow_retries,
            'max_overflow_retries_per_window':MAX_OVERFLOW_RETRIES,
            'final_loss_scale':float(scaler.get_scale()),
            'best_validation_loss':best_loss,'history':history,'seed':425,'epochs':epochs,
            'batch_size':batch_size,'accumulation':accumulation,'learning_rate':learning_rate,
            'weight_decay':.01,'max_length':512,'lora_rank':8,'lora_alpha':16,'lora_dropout':.05,
            'trainable_parameters':sum(p.numel() for p in trainable),'base_unchanged':base_unchanged,
            'initial_adapter_parameters_sha256':initial_hash,'best_adapter_parameters_sha256':best_hash,
            'weights_changed':True,'adapter_sha256':file_sha256(output/'best/adapter_model.safetensors'),
            'elapsed_seconds':perf_counter()-started,
            'epochs_completed':len(history),'early_stopping_patience':early_stopping_patience,
            'stopped_early':stop_reason=='validation_patience','stop_reason':stop_reason,
            'packages':{name:version(name) for name in ['torch','transformers','peft','tokenizers','safetensors']}}
    (output/'best/qwen_training.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
