"""SEA-LION expanded task data, saved targets versus original-target preservation."""
import argparse,gc,hashlib,json,random,time,traceback
from pathlib import Path
import torch
from peft import PeftModel
from replycoach.generate.sealion_data import encode_reply
from replycoach.generate.sealion_dual_v1 import STYLES,item_rows
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
from run_sealion_dual_v1 import ROOT,OUT,read,read_items,save_report,load_base,generate_pairs
from run_sealion_dialogue_v1 import validation,parameters,enable_training,update,step,save_adapter


def expanded_rows(expanded,original,validation_items,policy):
    if policy not in ['saved','preserved']:raise ValueError('Unknown target policy')
    original_map={r['id']:r for r in original};ids=[r['id'] for r in expanded]
    if len(ids)!=790 or len(set(ids))!=790 or len(original_map)!=190:raise ValueError('Invalid training counts')
    if not set(original_map)<=set(ids) or set(ids)&{r['id'] for r in validation_items}:raise ValueError('Invalid original coverage or split overlap')
    result=[]
    for r in expanded:
        if r['split']!='dev':raise ValueError('Only development training data allowed')
        if r['id'] in original_map:
            old=original_map[r['id']]
            if any(r[k]!=old[k] for k in ['session_memory','message']):raise ValueError('Original input changed')
        selected=original_map[r['id']] if policy=='preserved' and r['id'] in original_map else r
        if set(selected['replies'])!={'standard','local'}:raise ValueError('Expected exact paired styles')
        context=[t['text'] for t in r['session_memory']];speakers=[t['speaker'] for t in r['session_memory']]
        if any(s not in ['peer','user','other'] for s in speakers):raise ValueError('Invalid roles')
        for style in STYLES:
            target=selected['replies'][style.lower()]
            if not isinstance(target,str) or not target.strip():raise ValueError('Empty target')
            result.append(dict(item_id=r['id'],context=context,speakers=speakers,message=r['message']['text'],style=style,response=target))
    return result

def checked_training_rows(recipe,original,original_val,policy):
    path=ROOT/recipe.get('training_file','data/training_expansion/blanket600_v1/expanded.jsonl')
    raw=path.read_bytes()
    if recipe.get('require_original_team_exact') and not raw.startswith((ROOT/'data/dev/train.jsonl').read_bytes()):raise ValueError('Original team records must be preserved byte-for-byte')
    expanded=[json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    return expanded_rows(expanded,original,original_val,policy)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',choices=['saved','preserved'],required=True);args=ap.parse_args()
    if (OUT/'results.json').exists():raise FileExistsError('Fresh snapshot required')
    recipe=read('recipe.json');download=read('download_manifest.json')
    for p,h in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Frozen source/data mismatch: '+p)
    if recipe['revision']!=download['revision'] or recipe['weights_sha256']!=download['weights_sha256']:raise ValueError('Wrong base identity')
    if hashlib.sha256((ROOT/'public_adapter/adapter_model.safetensors').read_bytes()).hexdigest()!=recipe['public_adapter_sha256']:raise ValueError('Wrong public initialization adapter')
    original=read_items('train');original_val=read_items('validation')
    train=checked_training_rows(recipe,original,original_val,args.policy);val=item_rows(original_val)
    seed=recipe['training']['seed']
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
    if 'L40S' not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported():raise ValueError('L40S BF16 GPU required')
    report=dict(status='running',policy=args.policy,seed=seed,recipe=recipe,history=[],outputs={},default_changed=False,gpu=torch.cuda.get_device_name(0),quality_review='Pending; reference CE is not reply quality.')
    save_report(report);started=time.monotonic()
    try:
        tok=checkpoint_tokenizer(ROOT/'model');encoded=[encode_reply(r,tok,384) for r in train]
        valid={s:[encode_reply(r,tok,384) for r in val if r['style']==s] for s in STYLES}
        report['budgets']=dict(train_targets=len(encoded),validation_targets=len(val),max_train_tokens=max(len(e['input_ids']) for e in encoded),targets_truncated=False)
        base,problems=load_base();model=PeftModel.from_pretrained(base,ROOT/'public_adapter',is_trainable=True);report['strict_loading_problems']=problems
        enable_training(model);params=parameters(model);report['trainable_parameters']=sum(p.numel() for p in params)
        report['initial_validation']=validation(model,valid);save_report(report)
        torch.manual_seed(10000+seed);torch.cuda.manual_seed_all(10000+seed)
        optimizer=torch.optim.AdamW(params,lr=recipe['training']['learning_rate'],weight_decay=recipe['training']['weight_decay']);best=None;stale=0;updates=0
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,recipe['training']['max_epochs']+1):
            model.train();optimizer.zero_grad(set_to_none=True);order=torch.randperm(len(encoded),generator=torch.Generator().manual_seed(seed+epoch)).tolist();total=0.
            for index,which in enumerate(order):
                group=min(8,len(order)-(index//8)*8);total+=update(model,optimizer,params,encoded[which],group)
                if (index+1)%8==0 or index+1==len(order):step(optimizer,params);updates+=1
            scores=validation(model,valid);report['history'].append(dict(epoch=epoch,optimizer_updates=updates,train_example_mean_ce=total/len(order),validation=scores))
            if best is None or scores['selection_score']<best:
                best=scores['selection_score'];stale=0;save_adapter(model,tok,'best_adapter');report.update(best_epoch=epoch,best_score=best)
            else:stale+=1
            report.update(stale_epochs=stale,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            save_report(report);print(args.policy,'epoch',epoch,scores,flush=True)
            if stale>=recipe['training']['patience']:break
        del optimizer,params,model,base;gc.collect();torch.cuda.empty_cache()
        base,problems=load_base();model=PeftModel.from_pretrained(base,OUT/'best_adapter',is_trainable=False).eval()
        report['selected_validation']=validation(model,valid)
        for route,path in [('reused12','data/evaluation/sealion_dialogue_v1/inputs.json'),('reused20','data/evaluation/sealion_dual_v1/inputs.json')]:generate_pairs(model,tok,read(path),report,route)
        val_inputs=[dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text']) for r in original_val]
        generate_pairs(model,tok,val_inputs,report,'reused_validation')
        flags=[dict(route=k,item_id=x['item_id'],style=s,status=y['status']) for k,cases in report['outputs'].items() for x in cases for s,y in x['styles'].items() if y['status']!='complete']
        report.update(status='complete_with_flags' if flags else 'complete',flagged_outputs=flags,seconds=time.monotonic()-started);save_report(report)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started);save_report(report);raise
if __name__=='__main__':main()
