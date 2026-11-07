"""One full usable public pass with resumable LoRA/optimizer state, then task FT."""
import gc,hashlib,json,math,os,random,time,traceback
from pathlib import Path
from collections import Counter
import torch
from peft import LoraConfig,get_peft_model,PeftModel,get_peft_model_state_dict,set_peft_model_state_dict
from replycoach.generate.sealion_data import encode_reply
from replycoach.generate.sealion_dual_v1 import STYLES,item_rows
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
from run_sealion_dual_v1 import ROOT,OUT,read,save_report,read_items,load_base,generate_pairs
from run_sealion_dialogue_v1 import validation,parameters,enable_training,update,step,save_adapter
from run_sealion_expanded_v1 import expanded_rows
PUBLIC='data/public/sealion_full_v1'
def jsonl(p):return [json.loads(l) for l in (ROOT/p).read_text().splitlines() if l.strip()]
def resume_payload(model,opt,report):
    return dict(adapter={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()},optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),report=report)
def save_resume(model,opt,report):
    path=OUT/'public_resume.pt';tmp=OUT/'public_resume.tmp';torch.save(resume_payload(model,opt,report),tmp);tmp.replace(path)
def restore_adapter(model,state):
    expected=get_peft_model_state_dict(model)
    if set(expected)!=set(state) or any(expected[k].shape!=v.shape for k,v in state.items()):raise ValueError('Resume adapter shape/key mismatch')
    set_peft_model_state_dict(model,state)
    if any(not torch.equal(v.detach().cpu(),state[k]) for k,v in get_peft_model_state_dict(model).items()):raise ValueError('Resume adapter tensor mismatch')

def main():
    recipe=read('recipe.json');download=read('download_manifest.json')
    for p,h in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Frozen file mismatch '+p)
    if recipe['revision']!=download['revision'] or recipe['weights_sha256']!=download['weights_sha256']:raise ValueError('Wrong base identity')
    manifest=read(PUBLIC+'/manifest.json')
    if manifest['status']!='ready_exploratory':raise ValueError('Public sample inspection required')
    if (OUT/'results.json').exists() and read('artifacts/results.json').get('status') in ['complete','complete_with_flags']:raise FileExistsError('Completed experiment; do not overwrite')
    public=jsonl(PUBLIC+'/train.jsonl');pv=jsonl(PUBLIC+'/validation.jsonl')
    if len(public)!=manifest['train_rows'] or {r['group_id'] for r in public}&{r['group_id'] for r in pv}:raise ValueError('Public count/group overlap')
    original,ov=read_items('train'),read_items('validation')
    taskraw=jsonl('data/training_expansion/generated600_repair_v1/combined_train.jsonl')
    if not (ROOT/'data/training_expansion/generated600_repair_v1/combined_train.jsonl').read_bytes().startswith((ROOT/'data/dev/train.jsonl').read_bytes()):raise ValueError('Original team records changed')
    train=expanded_rows(taskraw,original,ov,'preserved');val=item_rows(ov)
    torch.set_num_threads(4);torch.manual_seed(425);torch.cuda.manual_seed_all(425);random.seed(425)
    if 'L40S' not in torch.cuda.get_device_name(0):raise ValueError('L40S required')
    OUT.mkdir(exist_ok=True);tok=checkpoint_tokenizer(ROOT/'model')
    pub=[encode_reply(r,tok,384) for r in public];validpub={c:[encode_reply(r,tok,384) for r in pv if r['corpus']==c] for c in ['cosem','dailydialog']}
    task=[encode_reply(r,tok,384) for r in train];valid={s:[encode_reply(r,tok,384) for r in val if r['style']==s] for s in STYLES}
    report=dict(status='running_public',recipe=recipe,public_manifest=manifest,public_history=[],task_history=[],outputs={},public_examples_seen=0,public_source_exposures={},public_optimizer_updates=0,public_loss_sum=0.,default_changed=False)
    report['budgets']=dict(public_train=len(pub),public_validation=len(pv),task_targets=len(task),task_validation=len(val),max_public_tokens=max(len(e['input_ids']) for e in pub),max_task_tokens=max(len(e['input_ids']) for e in task),no_truncation=True)
    try:
        model,problems=load_base();report['strict_loading_problems']=problems
        model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=.05,target_modules=['Wqkv','out_proj','up_proj','down_proj']))
        enable_training(model);params=parameters(model);report['trainable_parameters']=sum(p.numel() for p in params);opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01)
        path=OUT/'public_resume.pt'
        if path.exists():
            # This is a locally generated trusted checkpoint, never downloaded from an untrusted source.
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state['report']['recipe']!=recipe:raise ValueError('Resume recipe mismatch')
            progress=state['report']['public_examples_seen']
            if not 0<=progress<=len(pub) or (progress%16 and progress!=len(pub)):raise ValueError('Resume must be at optimizer boundary')
            restore_adapter(model,state['adapter']);opt.load_state_dict(state['optimizer']);report=state['report'];torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
            print('Resuming exact public prefix',report['public_examples_seen'],flush=True)
        else:
            scores=validation(model,validpub);report['public_history'].append(dict(examples_seen=0,validation=scores));save_resume(model,opt,report)
        save_report(report);order=torch.randperm(len(pub),generator=torch.Generator().manual_seed(1425)).tolist();seen=Counter(report['public_source_exposures']);opt.zero_grad(set_to_none=True);model.train()
        for index in range(report['public_examples_seen'],len(order)):
            count=index+1;which=order[index];group=min(16,len(order)-(index//16)*16)
            report['public_loss_sum']+=update(model,opt,params,pub[which],group);seen[public[which]['corpus']]+=1
            if count%16==0 or count==len(order):step(opt,params);report['public_optimizer_updates']+=1
            if count%10000==0 or count==len(order):
                scores=validation(model,validpub);report['public_history'].append(dict(examples_seen=count,validation=scores));print('Public',count,'/',len(pub),scores,flush=True);model.train()
            if count%4000==0 or count==len(order):
                report.update(public_examples_seen=count,public_source_exposures=dict(seen));save_resume(model,opt,report);save_report(report);print('Checkpoint',count,flush=True)
        report.update(public_examples_seen=len(pub),public_source_exposures=dict(seen),public_completed_full_pass=True,public_selection='Final full-pass adapter; not an earlier best validation checkpoint.');save_adapter(model,tok,'full_public_adapter');save_report(report)
        del opt,params,model;gc.collect();torch.cuda.empty_cache()
        base,problems=load_base();model=PeftModel.from_pretrained(base,OUT/'full_public_adapter',is_trainable=True);enable_training(model);params=parameters(model)
        torch.manual_seed(10425);torch.cuda.manual_seed_all(10425);report.update(status='running_task',task_history=[],outputs={});report['task_initial_validation']=validation(model,valid)
        opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);best=None;stale=0
        for epoch in range(1,9):
            model.train();opt.zero_grad(set_to_none=True);order=torch.randperm(len(task),generator=torch.Generator().manual_seed(425+epoch)).tolist();total=0.
            for index,which in enumerate(order):
                group=min(8,len(order)-(index//8)*8);total+=update(model,opt,params,task[which],group)
                if (index+1)%8==0 or index+1==len(order):step(opt,params)
            scores=validation(model,valid);report['task_history'].append(dict(epoch=epoch,train_example_mean_ce=total/len(task),validation=scores))
            if best is None or scores['selection_score']<best:
                best=scores['selection_score'];stale=0;report.update(best_epoch=epoch,best_score=best);save_adapter(model,tok,'best_adapter')
            else:stale+=1
            save_report(report);print('Task',epoch,scores,flush=True)
            if stale>=2:break
        del opt,params,model,base;gc.collect();torch.cuda.empty_cache();base,problems=load_base();model=PeftModel.from_pretrained(base,OUT/'best_adapter',is_trainable=False).eval();report['selected_validation']=validation(model,valid)
        for route,p in [('reused12','data/evaluation/sealion_dialogue_v1/inputs.json'),('reused20','data/evaluation/sealion_dual_v1/inputs.json')]:generate_pairs(model,tok,read(p),report,route)
        vi=[dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text']) for r in ov];generate_pairs(model,tok,vi,report,'reused_validation')
        flags=[dict(route=k,item_id=x['item_id'],style=s,status=y['status']) for k,cases in report['outputs'].items() for x in cases for s,y in x['styles'].items() if y['status']!='complete'];report.update(status='complete_with_flags' if flags else 'complete',flagged_outputs=flags);save_report(report)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());save_report(report);raise
if __name__=='__main__':main()
