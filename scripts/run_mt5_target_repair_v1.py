"""Matched target-repair experiment; autonomous generation quality is the goal."""
import gc,hashlib,json,random,time
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.style_control import mt5_input
from replycoach.generate.train_v5 import collate,epoch_batches,validation_loss
from replycoach.generate.v5_provider import V5Generator,DECODING
from replycoach.generate.rewrite_v5 import rewrite_standard

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'
DATA='data/training_expansion/mt5_grounding_repair_v1'


def read(p):return json.loads((ROOT/p).read_text())


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def source(row):return mt5_input('STANDARD',row['message'],row['context'],True,row['speakers'])


def rewrite_inference_rows(rows):
    # Shared encoder requires a target field even for generation. Never supply
    # validation references as that field.
    return [{**row,'response':''} for row in rows]


def training_rows(arm):
    return [json.loads(l) for l in (ROOT/f'{DATA}/{arm}.jsonl').read_text().splitlines()]


def verify_controls(control,repair,expected):
    if len(control)!=238 or len(repair)!=238:raise ValueError('Wrong training size')
    if len({r['item_id'] for r in control})!=238:raise ValueError('Duplicate IDs')
    changed=[]
    for a,b in zip(control,repair,strict=True):
        if any(a[k]!=b[k] for k in ['item_id','context','speakers','message','style']):raise ValueError('Unmatched controls')
        if a['style']!='STANDARD':raise ValueError('STANDARD only')
        if a['response']!=b['response']:changed.append(a['item_id'])
    if set(changed)!=set(expected) or len(changed)!=5:raise ValueError('Unexpected target changes')
    return changed


def original_validation():
    records=[json.loads(l) for l in (ROOT/'data/dev/validation.jsonl').read_text().splitlines()]
    return [dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],response=r['replies']['standard']) for r in records]


def encode(rows,tok):
    result=[]
    for r in rows:
        a=tok(source(r))['input_ids'];b=tok(r['response'])['input_ids']
        if len(a)>192 or len(b)>48:raise ValueError('No truncation: '+r['item_id'])
        result.append((a,b))
    return result


@torch.no_grad()
def generate(model,tok,rows):
    model.eval();result=[]
    for r in rows:
        ids=tok(source(r),return_tensors='pt')['input_ids'].cuda()
        if ids.shape[-1]>192:raise ValueError('Generation budget exceeded')
        tokens=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=48,**DECODING)
        text=tok.decode(tokens[0],skip_special_tokens=True)
        result.append(dict(item_id=r['item_id'],standard=text,standard_status='complete' if text.strip() else 'failed_empty_standard'))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise FileExistsError('Fresh report required')
    recipe=read('recipe.json')
    for p,h in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Frozen source mismatch: '+p)
    control,repair=training_rows('control'),training_rows('repair');verify_controls(control,repair,recipe['changed_target_ids'])
    vo=original_validation();vp=read('data/training_expansion/mt5_act_follow_v1/validation.json')
    old=read('data/evaluation/mt5_grounding_audit_v1/inputs.json');fresh=read('data/evaluation/mt5_target_repair_v1/inputs.json')
    if {r['item_id'] for r in control}&{r['item_id'] for r in vo+vp}:raise ValueError('Training/validation ID overlap')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    base=Path(recipe['base_checkpoint']);tok=AutoTokenizer.from_pretrained(base)
    encoded={k:encode(v,tok) for k,v in [('control',control),('repair',repair)]};eo,ep=encode(vo,tok),encode(vp,tok)
    data_map=dict(fresh=fresh,reused_diagnostics=old,original_validation=vo)
    report=dict(status='running',recipe=recipe,runs={},limitations=recipe['evaluation'])
    dump(OUT/'results.json',report)
    for seed in recipe['seeds']:
        for arm in ['control','repair']:
            started=time.monotonic();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
            model=AutoModelForSeq2SeqLM.from_pretrained(base).cuda();model.gradient_checkpointing_enable();model.config.use_cache=False
            opt=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
            def metrics():
                a=validation_loss(model,eo,2,tok.pad_token_id,'cuda');b=validation_loss(model,ep,2,tok.pad_token_id,'cuda')
                return dict(original_ce=a,paired_ce=b,selection_score=.5*(a+b))
            key=f'{arm}_s{seed}';initial=metrics();run=dict(arm=arm,seed=seed,history=[dict(epoch=0,**initial)],best_epoch=0,best_score=initial['selection_score'],outputs={})
            report['runs'][key]=run;best=OUT/key/'best';stale=0
            def save():
                model.save_pretrained(best);tok.save_pretrained(best)
                (best/'v5_input.json').write_text(json.dumps(dict(speaker_labels=True,joint=False,standard_only_diagnostic=True,max_source=192,max_target=48)))
            save();dump(OUT/'results.json',report)
            for epoch in range(1,recipe['max_epochs']+1):
                model.train();opt.zero_grad(set_to_none=True);batches=epoch_batches(len(encoded[arm]),2,seed,epoch);total=0.
                for bi,indices in enumerate(batches):
                    ids,mask,labels=collate([encoded[arm][i] for i in indices],tok.pad_token_id)
                    loss=model(input_ids=ids.cuda(),attention_mask=mask.cuda(),labels=labels.cuda()).loss
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
                    group=min(4,len(batches)-(bi//4)*4);(loss/group).backward();total+=float(loss.detach())
                    if (bi+1)%4==0 or bi+1==len(batches):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
                values=metrics();run['history'].append(dict(epoch=epoch,train_batch_mean_loss=total/len(batches),**values))
                if values['selection_score']<run['best_score']:
                    run.update(best_score=values['selection_score'],best_epoch=epoch);save();stale=0
                else:stale+=1
                dump(OUT/'results.json',report);print(key,'epoch',epoch,values,flush=True)
                if stale>=recipe['patience']:break
            del opt,model;gc.collect();torch.cuda.empty_cache()
            model=AutoModelForSeq2SeqLM.from_pretrained(best).cuda();model.config.use_cache=True
            run['selected_validation']=metrics()
            for name,data in data_map.items():
                run['outputs'][name]=generate(model,tok,data);dump(OUT/'results.json',report)
            run['training_and_standard_seconds']=time.monotonic()-started
            del model;gc.collect();torch.cuda.empty_cache()
    local=V5Generator(Path(recipe['rewrite_checkpoint']),device='cuda')
    for run in report['runs'].values():
        for name,items in run['outputs'].items():
            eligible=[]
            for row,item in zip(data_map[name],items,strict=True):
                if row['item_id']!=item['item_id']:raise ValueError('Rewrite ID mismatch')
                anchor=item['standard']
                if not anchor.strip():item['rewrite_status']='blocked_empty_standard';continue
                text=mt5_input('LOCAL_REWRITE',row['message'],row['context'],True,row['speakers'],standard_reply=anchor)
                if len(local.tokenizer(text)['input_ids'])>local.max_source:item['rewrite_status']='blocked_source_budget';continue
                eligible.append((row,item))
            for start in range(0,len(eligible),4):
                part=eligible[start:start+4];anchors=[item['standard'] for _,item in part]
                outputs=rewrite_standard(local.model,local.tokenizer,rewrite_inference_rows([row for row,_ in part]),anchors,local.max_source,local.max_target,'cuda',DECODING,True)
                for (_,item),anchor,text in zip(part,anchors,outputs,strict=True):
                    item.update(local_standard_anchor=anchor,local=text,rewrite_status='complete' if text.strip() else 'failed_empty_local')
                dump(OUT/'results.json',report)
    failures=[dict(run=k,route=route,item_id=x['item_id'],status=x['rewrite_status']) for k,run in report['runs'].items() for route,items in run['outputs'].items() for x in items if x['rewrite_status']!='complete']
    report.update(status='complete_with_failures' if failures else 'complete',failed_stages=failures)
    dump(OUT/'results.json',report);print('Completed matched target repair',report['status'],flush=True)

if __name__=='__main__':main()
