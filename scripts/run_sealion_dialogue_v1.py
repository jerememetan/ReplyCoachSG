"""Public dialogue LoRA adaptation -> original dual-style task fine-tuning."""
import gc,json,math,random,time,traceback
from collections import Counter
from pathlib import Path
import torch
from peft import LoraConfig,get_peft_model,PeftModel
from replycoach.generate.sealion_data import encode_reply
from replycoach.generate.sealion_dual_v1 import STYLES,item_rows,reply_loss
from run_sealion_dual_v1 import ROOT,OUT,read,save_report,read_items,batch,load_base,generate_pairs
from replycoach.generate.sealion_v1 import checkpoint_tokenizer

PUBLIC='data/public/sealion_dialogue_v1'


@torch.no_grad()
def validation(model,groups):
    model.eval();values={}
    for name,examples in groups.items():
        total=0.;tokens=0
        for e in examples:
            data=batch(e);labels=data.pop('labels');out=model(**data,use_cache=False)
            loss=reply_loss(out.logits,labels);n=int(labels[:,1:].ne(-100).sum())
            total+=float(loss)*n;tokens+=n
        if not tokens:raise ValueError('Empty validation slice')
        values[name]=dict(token_ce=total/tokens,reply_tokens=tokens,examples=len(examples))
    return dict(groups=values,selection_score=sum(v['token_ce'] for v in values.values())/len(values))


def jsonl(path):return [json.loads(l) for l in (ROOT/path).read_text().splitlines() if l.strip()]


def parameters(model):
    result=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not result or any('lora_' not in n for n,_ in result):raise ValueError('Only LoRA adapters may train')
    return [p for _,p in result]


def enable_training(model):
    model.gradient_checkpointing_enable();model.enable_input_require_grads();model.config.use_cache=False


def update(model,optimizer,params,e,scale):
    data=batch(e);labels=data.pop('labels');out=model(**data,use_cache=False)
    loss=reply_loss(out.logits,labels)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    (loss/scale).backward();value=float(loss.detach())
    return value


def step(optimizer,params):
    norm=torch.nn.utils.clip_grad_norm_(params,1.)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
    optimizer.step();optimizer.zero_grad(set_to_none=True)


def save_adapter(model,tok,name):
    model.save_pretrained(OUT/name);tok.save_pretrained(OUT/name)


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise FileExistsError('Fresh run required')
    recipe=read('recipe.json');download=read('download_manifest.json')
    import hashlib
    for path,digest in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen file mismatch: '+path)
    if download['revision']!=recipe['revision'] or download['weights_sha256']!=recipe['weights_sha256']:raise ValueError('Base identity mismatch')
    manifest=read(PUBLIC+'/manifest.json')
    if manifest['status']!='ready_exploratory':raise ValueError('Public sample inspection required')
    for name,h in manifest['files_sha256'].items():
        if hashlib.sha256((ROOT/PUBLIC/name).read_bytes()).hexdigest()!=h:raise ValueError('Public file mismatch: '+name)
    public_train=jsonl(PUBLIC+'/train.jsonl');public_val=jsonl(PUBLIC+'/validation.jsonl')
    if len(public_train)!=7994 or len(public_val)!=400:raise ValueError('Frozen public counts changed')
    if {r['group_id'] for r in public_train}&{r['group_id'] for r in public_val}:raise ValueError('Public group leakage')
    original_train,original_val=read_items('train'),read_items('validation')
    task_train,task_val=item_rows(original_train),item_rows(original_val)
    if len(task_train)!=380 or len(task_val)!=100:raise ValueError('Task dual-style counts changed')
    if {r['item_id'] for r in task_train}&{r['item_id'] for r in task_val}:raise ValueError('Task split overlap')
    fresh=read('data/evaluation/sealion_dialogue_v1/inputs.json')
    reused=read('data/evaluation/sealion_dual_v1/inputs.json')
    torch.set_num_threads(4);torch.manual_seed(425);torch.cuda.manual_seed_all(425);random.seed(425)
    if '3090' not in torch.cuda.get_device_name(0):raise ValueError('RTX3090 required')
    report=dict(status='running',recipe=recipe,public_manifest=manifest,public_history=[],task_history=[],outputs={},default_changed=False,quality_review='Pending; generated outputs are not quality scores.')
    save_report(report);started=time.monotonic()
    try:
        tok=checkpoint_tokenizer(ROOT/'model')
        pub=[encode_reply(r,tok,384) for r in public_train]
        pub_val={c:[encode_reply(r,tok,384) for r in public_val if r['corpus']==c] for c in ['cosem','dailydialog']}
        task=[encode_reply(r,tok,384) for r in task_train]
        valid={s:[encode_reply(r,tok,384) for r in task_val if r['style']==s] for s in STYLES}
        report['budgets']=dict(public_train=len(pub),public_validation=sum(map(len,pub_val.values())),project_train=len(task),project_validation=sum(map(len,valid.values())),max_public_tokens=max(len(e['input_ids']) for e in pub),max_project_tokens=max(len(e['input_ids']) for e in task))
        model,problems=load_base();report['strict_loading_problems']=problems
        model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=.05,target_modules=recipe['lora']['target_modules']))
        enable_training(model);params=parameters(model)
        report['trainable_parameters']=sum(p.numel() for p in params)
        # Restore this RNG point before task training so its sampling/dropout
        # policy matches the original project-only setup as closely as possible.
        task_cpu_rng=torch.get_rng_state();task_cuda_rng=torch.cuda.get_rng_state_all()
        initial_public=validation(model,pub_val);public_best=initial_public['selection_score']
        report['public_history'].append(dict(examples_seen=0,optimizer_updates=0,validation=initial_public))
        report['public_best_examples_seen']=0;report['public_best_score']=public_best
        save_adapter(model,tok,'public_best_adapter');save_report(report)
        optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01)
        order=torch.randperm(len(pub),generator=torch.Generator().manual_seed(1425)).tolist()
        optimizer.zero_grad(set_to_none=True);model.train();updates=0;loss_sum=0.;source_seen=Counter()
        torch.cuda.reset_peak_memory_stats()
        for index,which in enumerate(order):
            count=index+1;group=min(16,len(order)-(index//16)*16)
            loss_sum+=update(model,optimizer,params,pub[which],group);source_seen[public_train[which]['corpus']]+=1
            if count%16==0 or count==len(order):step(optimizer,params);updates+=1
            if count%2000==0 or count==len(order):
                scores=validation(model,pub_val)
                report['public_history'].append(dict(examples_seen=count,optimizer_updates=updates,train_example_mean_ce=loss_sum/count,validation=scores))
                if scores['selection_score']<public_best:
                    public_best=scores['selection_score'];report.update(public_best_examples_seen=count,public_best_score=public_best);save_adapter(model,tok,'public_best_adapter')
                report.update(public_examples_seen=count,public_optimizer_updates=updates,public_source_exposures=dict(source_seen),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                save_report(report);print('Public adaptation',count,'/',len(pub),scores,flush=True);model.train()
        del optimizer,params,model;gc.collect();torch.cuda.empty_cache()
        base,problems=load_base();model=PeftModel.from_pretrained(base,OUT/'public_best_adapter',is_trainable=True)
        enable_training(model);params=parameters(model)
        torch.set_rng_state(task_cpu_rng);torch.cuda.set_rng_state_all(task_cuda_rng)
        report['task_initial_validation']=validation(model,valid);save_report(report)
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
        best=None;best_epoch=None;stale=0;updates=0
        for epoch in range(1,13):
            model.train();optimizer.zero_grad(set_to_none=True)
            order=torch.randperm(len(task),generator=torch.Generator().manual_seed(425+epoch)).tolist();total=0.
            for index,which in enumerate(order):
                group=min(8,len(order)-(index//8)*8);total+=update(model,optimizer,params,task[which],group)
                if (index+1)%8==0 or index+1==len(order):step(optimizer,params);updates+=1
            scores=validation(model,valid)
            report['task_history'].append(dict(epoch=epoch,optimizer_updates=updates,train_example_mean_ce=total/len(task),validation=scores))
            if best is None or scores['selection_score']<best:
                best=scores['selection_score'];best_epoch=epoch;stale=0
                save_adapter(model,tok,'best_adapter');report.update(best_task_epoch=best_epoch,best_task_score=best)
            else:stale+=1
            report.update(task_stale_epochs=stale,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            save_report(report);print('Project epoch',epoch,scores,'selected',best_epoch,flush=True)
            if stale>=2:break
        report['task_optimizer_updates']=updates
        del optimizer,params,model,base;gc.collect();torch.cuda.empty_cache()
        base,problems=load_base();model=PeftModel.from_pretrained(base,OUT/'best_adapter',is_trainable=False).eval()
        report['selected_validation']=validation(model,valid);save_report(report)
        generate_pairs(model,tok,fresh,report,'selected_new_development')
        generate_pairs(model,tok,reused,report,'selected_reused20')
        val_inputs=[dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text']) for r in original_val]
        generate_pairs(model,tok,val_inputs,report,'selected_reused_validation')
        del model,base;gc.collect();torch.cuda.empty_cache()
        base,problems=load_base();model=PeftModel.from_pretrained(base,ROOT/'project_only_adapter',is_trainable=False).eval()
        generate_pairs(model,tok,fresh,report,'project_only_new_development')
        report['project_only_validation_reproduction']=validation(model,valid)
        prior=read('project_only_results.json')
        report['project_only_reference']=dict(best_epoch=prior['best_trained_epoch'],validation=prior['selected_validation'],reused20=prior['outputs']['selected_fresh'],note='Same original380 target-only trained SEA-LION adapter; original paired outputs preserved. Public adaptation adds data/compute, not a matched-FLOP ablation.')
        flags=[dict(route=k,item_id=c['item_id'],style=s,status=x['status']) for k,cs in report['outputs'].items() for c in cs for s,x in c['styles'].items() if x['status']!='complete']
        report.update(status='complete_with_flags' if flags else 'complete',flagged_outputs=flags,seconds=time.monotonic()-started)
        save_report(report);print('Completed staged SEA-LION dialogue adaptation',report['status'],flush=True)
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started);save_report(report);raise

if __name__=='__main__':main()
