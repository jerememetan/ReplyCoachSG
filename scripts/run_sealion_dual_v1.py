"""Train one dual-style SEA-LION adapter; keep reference and fresh evaluations apart."""
import gc,hashlib,json,math,random,time,traceback
from pathlib import Path
import torch
from peft import LoraConfig,get_peft_model,PeftModel
from replycoach.generate.sealion_v1 import SEALionCausalLM,config_for_checkpoint,checkpoint_tokenizer,checkpoint_loading_problems
from replycoach.generate.sealion_data import encode_reply
from replycoach.generate.sealion_dual_v1 import STYLES,item_rows,reply_loss,generate_reply

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'


def read(p):return json.loads((ROOT/p).read_text())


def save_report(report):
    OUT.mkdir(exist_ok=True);temp=OUT/'results.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(OUT/'results.json')


def read_items(split):return [json.loads(l) for l in (ROOT/f'data/dev/{split}.jsonl').read_text().splitlines()]


def batch(example):return {k:torch.tensor([example[k]],device='cuda') for k in ['input_ids','attention_mask','labels']}


@torch.no_grad()
def validation(model,encoded):
    model.eval();scores={}
    for style,examples in encoded.items():
        summed=0.;tokens=0
        for example in examples:
            data=batch(example);labels=data.pop('labels')
            output=model(**data,use_cache=False)
            loss=reply_loss(output.logits,labels);count=int(labels[:,1:].ne(-100).sum())
            summed+=float(loss)*count;tokens+=count
        scores[style]=dict(token_ce=summed/tokens,reply_tokens=tokens,examples=len(examples))
    return dict(by_style=scores,selection_score=sum(scores[s]['token_ce'] for s in STYLES)/2)


def generate_pairs(model,tok,rows,report,key):
    model.eval();report['outputs'][key]=[]
    for row in rows:
        case=dict(item_id=row['item_id'],styles={})
        for style in STYLES:case['styles'][style]=generate_reply(model,tok,row,style)
        report['outputs'][key].append(case);save_report(report)
    return report['outputs'][key]


def load_base():
    model,info=SEALionCausalLM.from_pretrained(ROOT/'model',config=config_for_checkpoint(ROOT/'model'),dtype=torch.bfloat16,device_map={'':0},local_files_only=True,output_loading_info=True)
    problems=checkpoint_loading_problems(info)
    if any(problems.values()) or any(p.is_meta for p in model.parameters()):raise ValueError('Base checkpoint loading mismatch: '+str(problems))
    return model,problems


def mt5_comparison(rows,report):
    """Same fresh cases, existing experimental English->LOCAL pipeline, no retraining."""
    from replycoach.generate.v5_provider import V5Generator,DECODING
    from replycoach.generate.train_v5 import generate_replies
    from replycoach.generate.rewrite_v5 import rewrite_standard
    from replycoach.generate.style_control import mt5_input
    standard_path=Path('/common/home/users/h/hm.teoh.2024/ReplyCoachSG_standard_val_v1/artifacts/full_lr3e4/best')
    local_path=Path('/common/home/users/h/hm.teoh.2024/ReplyCoachSG_hybrid_inference_v1/checkpoints/local_rewrite')
    comp=dict(status='running',standard_checkpoint=str(standard_path),local_checkpoint=str(local_path),scope='Existing experimental mT5 pipeline, not app default or matched-data/parameter/pretraining control. STANDARD checkpoint used790 project training rows; SEA-LION run uses190 original conversations.',cases=[])
    report['mt5_comparison']=comp;save_report(report)
    model=V5Generator(standard_path,device='cuda')
    candidates=[]
    for row in rows:
        c=dict(item_id=row['item_id']);comp['cases'].append(c)
        text=mt5_input('STANDARD',row['message'],row['context'],True,row['speakers'])
        if len(model.tokenizer(text)['input_ids'])>model.max_source:c['standard_status']='blocked_source_budget'
        else:candidates.append((row,c))
    for start in range(0,len(candidates),4):
        part=candidates[start:start+4]
        inputs=[{**row,'response':''} for row,_ in part]
        texts=generate_replies(model.model,model.tokenizer,inputs,'STANDARD',model.max_source,model.max_target,'cuda',DECODING,True)
        for (_,c),text in zip(part,texts,strict=True):c.update(standard=text,standard_status='complete' if text.strip() else 'failed_empty')
        save_report(report)
    del model;gc.collect();torch.cuda.empty_cache()
    local=V5Generator(local_path,device='cuda');eligible=[]
    for row,c in zip(rows,comp['cases'],strict=True):
        anchor=c.get('standard','')
        if not anchor.strip():c['local_status']='blocked_standard_failure';continue
        text=mt5_input('LOCAL_REWRITE',row['message'],row['context'],True,row['speakers'],standard_reply=anchor)
        if len(local.tokenizer(text)['input_ids'])>local.max_source:c['local_status']='blocked_source_budget';continue
        eligible.append((row,c))
    for start in range(0,len(eligible),4):
        part=eligible[start:start+4];anchors=[c['standard'] for _,c in part]
        texts=rewrite_standard(local.model,local.tokenizer,[{**row,'response':''} for row,_ in part],anchors,local.max_source,local.max_target,'cuda',DECODING,True)
        for (_,c),anchor,text in zip(part,anchors,texts,strict=True):c.update(local=text,local_standard_anchor=anchor,local_status='complete' if text.strip() else 'failed_empty')
        save_report(report)
    comp['status']='complete';save_report(report)


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise FileExistsError('Fresh snapshot required')
    recipe=read('recipe.json');download=read('download_manifest.json')
    if download['revision']!=recipe['revision'] or download['weights_sha256']!=recipe['weights_sha256']:raise ValueError('Base identity mismatch')
    for p,digest in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source mismatch: '+p)
    torch.set_num_threads(4);torch.manual_seed(recipe['seed']);torch.cuda.manual_seed_all(recipe['seed']);random.seed(recipe['seed'])
    if not torch.cuda.is_bf16_supported() or '3090' not in torch.cuda.get_device_name(0):raise ValueError('RTX3090 BF16 environment required')
    original_train,original_val=read_items('train'),read_items('validation')
    train,val=item_rows(original_train),item_rows(original_val)
    if len(train)!=380 or len(val)!=100 or {r['item_id'] for r in train}&{r['item_id'] for r in val}:raise ValueError('Dual-style split counts/overlap invalid')
    fresh=read('data/evaluation/sealion_dual_v1/inputs.json')
    report=dict(status='running',recipe=recipe,history=[],outputs={},default_changed=False,quality_review='Not yet performed; generated replies are not scores.')
    save_report(report);started=time.monotonic()
    try:
        tok=checkpoint_tokenizer(ROOT/'model')
        encoded=[encode_reply(r,tok,384) for r in train]
        valid={style:[encode_reply(r,tok,384) for r in val if r['style']==style] for style in STYLES}
        report['data_verified']=dict(train_targets=len(encoded),validation_targets=sum(map(len,valid.values())),max_train_tokens=max(len(r['input_ids']) for r in encoded),max_validation_tokens=max(len(r['input_ids']) for rows in valid.values() for r in rows),targets_preserved=True)
        model,problems=load_base();report['strict_loading_problems']=problems
        model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=.05,target_modules=recipe['lora']['target_modules']))
        model.gradient_checkpointing_enable();model.enable_input_require_grads();model.config.use_cache=False
        params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not params or any('lora_' not in n for n,_ in params):raise ValueError('Unexpected unfrozen base parameters')
        report['trainable_parameters']=sum(p.numel() for _,p in params)
        report['baseline_validation']=validation(model,valid);save_report(report)
        generate_pairs(model,tok,fresh,report,'step0_base_fresh')
        # Baseline is a diagnostic only. The selected candidate must have at
        # least one supervised epoch to satisfy the trained-model requirement.
        optimizer=torch.optim.AdamW([p for _,p in params],lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
        best=None;best_epoch=None;stale=0;updates=0
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,recipe['max_epochs']+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            order=torch.randperm(len(encoded),generator=torch.Generator().manual_seed(recipe['seed']+epoch)).tolist()
            loss_sum=0.
            for index,which in enumerate(order):
                data=batch(encoded[which]);labels=data.pop('labels')
                out=model(**data,use_cache=False);loss=reply_loss(out.logits,labels)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                group=min(8,len(order)-(index//8)*8)
                (loss/group).backward();loss_sum+=float(loss.detach())
                if (index+1)%8==0 or index+1==len(order):
                    norm=torch.nn.utils.clip_grad_norm_([p for _,p in params],1.)
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite LoRA gradient norm')
                    optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                del out,loss,data
            scores=validation(model,valid)
            entry=dict(epoch=epoch,optimizer_updates=updates,train_example_mean_ce=loss_sum/len(order),validation=scores)
            report['history'].append(entry)
            if best is None or scores['selection_score']<best:
                best=scores['selection_score'];best_epoch=epoch;stale=0
                model.save_pretrained(OUT/'best_adapter');tok.save_pretrained(OUT/'best_adapter')
                report.update(best_trained_epoch=best_epoch,best_trained_score=best)
            else:stale+=1
            report.update(stale_epochs=stale,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            save_report(report);print('Epoch',epoch,'validation',scores,'best trained epoch',best_epoch,flush=True)
            if stale>=recipe['patience']:break
        report['optimizer_updates']=updates
        del optimizer,params,model;gc.collect();torch.cuda.empty_cache()
        base,problems=load_base();model=PeftModel.from_pretrained(base,OUT/'best_adapter',is_trainable=False).eval()
        report['selected_validation']=validation(model,valid)
        report['selected_improves_baseline_ce']=report['selected_validation']['selection_score']<report['baseline_validation']['selection_score']
        save_report(report)
        generate_pairs(model,tok,fresh,report,'selected_fresh')
        val_inputs=[dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text']) for r in original_val]
        generate_pairs(model,tok,val_inputs,report,'selected_reused_validation')
        del model,base;gc.collect();torch.cuda.empty_cache()
        mt5_comparison(fresh,report)
        statuses=[dict(route=key,item_id=c['item_id'],style=style,status=x['status']) for key,cs in report['outputs'].items() for c in cs for style,x in c['styles'].items() if x['status']!='complete']
        report.update(status='complete_with_flags' if statuses else 'complete',flagged_outputs=statuses,seconds=time.monotonic()-started)
        save_report(report);print('Completed dual-style SEA-LION run',report['status'],flush=True)
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
        save_report(report);raise

if __name__=='__main__':main()
