"""RTX3090 LoRA feasibility check only: no quality ranking or model promotion."""
import hashlib,json,platform,time,traceback
from pathlib import Path
import torch
import transformers,peft
from peft import get_peft_model,LoraConfig
from replycoach.generate.sealion_v1 import SEALionCausalLM,config_for_checkpoint,checkpoint_tokenizer,checkpoint_loading_problems
from replycoach.generate.sealion_data import source,encode_reply,pad_example

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'


def write(report):
    OUT.mkdir(exist_ok=True);tmp=OUT/'probe.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(OUT/'probe.json')


def main():
    if (OUT/'probe.json').exists():raise FileExistsError('Fresh probe report required')
    recipe=json.loads((ROOT/'recipe.json').read_text());download=json.loads((ROOT/'download_manifest.json').read_text())
    if download['weights_sha256']!=recipe['weights_sha256'] or download['revision']!=recipe['revision']:raise ValueError('Download manifest mismatch')
    for path,digest in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source mismatch: '+path)
    torch.set_num_threads(4);torch.manual_seed(425);torch.cuda.manual_seed_all(425)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    device=torch.cuda.get_device_properties(0)
    if '3090' not in device.name:raise ValueError('Expected allocated RTX3090')
    if not torch.cuda.is_bf16_supported():raise ValueError('BF16 support required')
    report=dict(status='running',purpose='Compatibility/memory probe, not reply-quality training or final model',recipe=recipe,download=download,gpu=dict(name=device.name,total_bytes=device.total_memory,free_bytes_at_start=torch.cuda.mem_get_info()[0]),versions=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__),steps=[])
    write(report)
    started=time.monotonic()
    try:
        tok=checkpoint_tokenizer(ROOT/'model');config=config_for_checkpoint(ROOT/'model')
        model,info=SEALionCausalLM.from_pretrained(ROOT/'model',config=config,dtype=torch.bfloat16,device_map={'':0},local_files_only=True,output_loading_info=True)
        problems=checkpoint_loading_problems(info)
        if any(problems.values()):raise ValueError('Checkpoint did not load exactly: '+str(problems))
        if any(p.is_meta for p in model.parameters()):raise ValueError('Unmaterialized weights')
        report['loading']=dict(parameter_count=sum(p.numel() for p in model.parameters()),architecture=type(model).__name__,learned_positions=model.transformer.learned_pos_emb,alibi=model.transformer.alibi,loading_problems=problems,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
        model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=.05,target_modules=['Wqkv','out_proj','up_proj','down_proj']))
        model.gradient_checkpointing_enable();model.enable_input_require_grads();model.config.use_cache=False
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n for n,_ in trainable):raise ValueError('Unexpected trainable base weights')
        report['trainable_parameters']=sum(p.numel() for _,p in trainable)
        raw=[json.loads(l) for l in (ROOT/'data/dev/train.jsonl').read_text().splitlines()]
        rows=[dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],response=r['replies']['local']) for r in raw]
        encoded=[(r,encode_reply(r,tok,384)) for r in rows]
        encoded.sort(key=lambda pair:len(pair[1]['input_ids']),reverse=True)
        report['token_budget']=dict(rows_verified=len(encoded),max_actual_tokens=len(encoded[0][1]['input_ids']),stress_padded_tokens=384,microbatch_size=1,eos_id=tok.eos_token_id,pad_id=tok.pad_token_id,objective='LOCAL reply-only causal loss; prompt/padding masked and real EOS retained')
        optimizer=torch.optim.AdamW([p for _,p in trainable],lr=1e-4,weight_decay=.01)
        before={n:p.detach().cpu().clone() for n,p in trainable}
        torch.cuda.reset_peak_memory_stats();model.train()
        for step,(row,example) in enumerate(encoded[:4],1):
            sample=pad_example(example,tok.pad_token_id,384)
            data={k:torch.tensor([v],device='cuda') for k,v in sample.items()}
            optimizer.zero_grad(set_to_none=True);out=model(**data,use_cache=False)
            if not torch.isfinite(out.loss):raise FloatingPointError('Nonfinite loss')
            out.loss.backward()
            grads=[p.grad for _,p in trainable if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):raise FloatingPointError('Missing/nonfinite LoRA gradients')
            if any(p.grad is not None for n,p in model.named_parameters() if 'lora_' not in n):raise ValueError('Frozen base received gradients')
            norm=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite norm')
            optimizer.step();torch.cuda.synchronize()
            report['steps'].append(dict(step=step,item_id=row['item_id'],loss=float(out.loss.detach()),grad_norm=float(norm),allocated_bytes=torch.cuda.memory_allocated(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            write(report);print('Probe step',step,'loss',float(out.loss.detach()),'peak GiB',torch.cuda.max_memory_allocated()/2**30,flush=True)
            del out,data
        changed=sum(not torch.equal(before[n],p.detach().cpu()) for n,p in trainable)
        if not changed:raise ValueError('No adapter weights changed')
        report['changed_adapter_tensors']=changed
        model.save_pretrained(OUT/'smoke_adapter');tok.save_pretrained(OUT/'smoke_adapter')
        # Short cache-free greedy continuation checks inference compatibility.
        # This four-step adapter is NOT assessed or presented as a trained model.
        model.eval();prompt=source(encoded[0][0]);tokens=torch.tensor([tok.encode(prompt,add_special_tokens=False)],device='cuda');generated=[]
        with torch.no_grad():
            for _ in range(4):
                logits=model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False).logits[:,-1,:]
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite inference logits')
                next_id=int(logits.argmax(-1));generated.append(next_id)
                tokens=torch.cat([tokens,torch.tensor([[next_id]],device='cuda')],dim=1)
                if next_id==tok.eos_token_id:break
        report.update(status='complete',feasibility_passed=True,seconds=time.monotonic()-started,continuation_tokens=len(generated),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),quality_evaluated=False,default_changed=False)
        write(report);print('SEA-LION3B compatibility and VRAM probe passed',flush=True)
    except Exception as exc:
        report.update(status='failed',feasibility_passed=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
        write(report);raise

if __name__=='__main__':main()
