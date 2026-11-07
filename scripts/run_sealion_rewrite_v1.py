"""Train separate anchored LOCAL renderer; original English adapter stays fixed."""
import gc,hashlib,json,random,time,traceback
import torch
from peft import PeftModel
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
from replycoach.generate.sealion_rewrite_v1 import encode_rewrite,generate_rewrite
from run_sealion_dual_v1 import ROOT,OUT,read,save_report,load_base
from run_sealion_dialogue_v1 import validation,parameters,enable_training,update,step,save_adapter
DATA='data/training_expansion/generated600_repair_v1/combined_train.jsonl'

def pair_rows(records,allowed):
    if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate source IDs')
    if not allowed<={r['id'] for r in records}:raise ValueError('Audit/source coverage mismatch')
    return [dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],standard_anchor=r['replies']['standard'],response=r['replies']['local']) for r in records if r['id'] in allowed]

def main():
    if (OUT/'results.json').exists():raise FileExistsError('Fresh snapshot required')
    recipe=read('recipe.json');download=read('download_manifest.json')
    for p,h in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Frozen source mismatch '+p)
    if recipe['revision']!=download['revision'] or recipe['weights_sha256']!=download['weights_sha256']:raise ValueError('Wrong base')
    for name,h in recipe['initial_adapter_sha256'].items():
        if hashlib.sha256((ROOT/'initial_adapter'/name).read_bytes()).hexdigest()!=h:raise ValueError('Initial adapter mismatch '+name)
    if not (ROOT/DATA).read_bytes().startswith((ROOT/'data/dev/train.jsonl').read_bytes()):raise ValueError('Team originals changed')
    audit=read('data/training_expansion/sealion_rewrite_v1/pair_audit.json')
    train_ids={x['id'] for x in audit['items'] if x['split']=='train' and x['eligible']};val_ids={x['id'] for x in audit['items'] if x['split']=='validation' and x['eligible']}
    if train_ids&val_ids or len(train_ids)!=745 or len(val_ids)!=42:raise ValueError('Audit split/count error')
    train=pair_rows([json.loads(l) for l in (ROOT/DATA).read_text().splitlines()],train_ids)
    val=pair_rows([json.loads(l) for l in (ROOT/'data/dev/validation.jsonl').read_text().splitlines()],val_ids)
    evaluation=read('data/evaluation/sealion_rewrite_v1/inputs.json')
    torch.set_num_threads(4);torch.manual_seed(425);torch.cuda.manual_seed_all(425);random.seed(425)
    if 'L40S' not in torch.cuda.get_device_name(0):raise ValueError('L40S required')
    report=dict(status='running',recipe=recipe,pair_audit_counts=audit['counts'],history=[],outputs={},default_changed=False,english_generator_changed=False)
    save_report(report);started=time.monotonic()
    try:
        tok=checkpoint_tokenizer(ROOT/'model');encoded=[encode_rewrite(r,tok) for r in train];valid={'LOCAL_REWRITE':[encode_rewrite(r,tok) for r in val]}
        report['budgets']=dict(train=len(encoded),validation=len(val),max_train_tokens=max(len(x['input_ids']) for x in encoded),no_truncation=True)
        base,problems=load_base();report['strict_loading_problems']=problems;model=PeftModel.from_pretrained(base,ROOT/'initial_adapter',is_trainable=True)
        enable_training(model);params=parameters(model);report['trainable_parameters']=sum(p.numel() for p in params)
        report['initial_validation']=validation(model,valid);save_report(report)
        opt=torch.optim.AdamW(params,lr=3e-5,weight_decay=.01);best=None;stale=0;updates=0
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,9):
            model.train();opt.zero_grad(set_to_none=True);order=torch.randperm(len(encoded),generator=torch.Generator().manual_seed(425+epoch)).tolist();total=0.
            for index,which in enumerate(order):
                group=min(8,len(order)-(index//8)*8);total+=update(model,opt,params,encoded[which],group)
                if (index+1)%8==0 or index+1==len(order):step(opt,params);updates+=1
            scores=validation(model,valid);report['history'].append(dict(epoch=epoch,optimizer_updates=updates,train_example_mean_ce=total/len(order),validation=scores))
            if best is None or scores['selection_score']<best:
                best=scores['selection_score'];stale=0;report.update(best_epoch=epoch,best_score=best);save_adapter(model,tok,'best_adapter')
            else:stale+=1
            report.update(stale_epochs=stale,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save_report(report);print('Rewrite epoch',epoch,scores,flush=True)
            if stale>=2:break
        del opt,params,model,base;gc.collect();torch.cuda.empty_cache()
        base,problems=load_base();model=PeftModel.from_pretrained(base,OUT/'best_adapter',is_trainable=False).eval()
        report['selected_validation']=validation(model,valid);save_report(report)
        for route,field in [('actual_english','actual_anchor'),('diagnostic_english','diagnostic_anchor')]:
            report['outputs'][route]=[]
            for row in evaluation:
                if field not in row:continue
                anchor=row[field];output=generate_rewrite(model,tok,row,anchor)
                if output['standard_anchor']!=anchor:raise ValueError('Anchor mismatch')
                report['outputs'][route].append(dict(item_id=row['item_id'],standard_anchor=anchor,direct_local=row['direct_local'],rewrite=output));save_report(report)
        flags=[dict(route=k,item_id=x['item_id'],status=x['rewrite']['status']) for k,cases in report['outputs'].items() for x in cases if x['rewrite']['status']!='complete']
        report.update(status='complete_with_flags' if flags else 'complete',flagged_outputs=flags,seconds=time.monotonic()-started,quality_review='Pending: assess fidelity to anchor, contextual usefulness and naturalness separately. Actual anchors may contradict context. Assistant diagnostic anchors are not human gold.');save_report(report)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started);save_report(report);raise
if __name__=='__main__':main()
