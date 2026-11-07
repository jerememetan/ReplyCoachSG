"""Matched mT5 direct dual-style arms with optional cleaned public adaptation."""
import argparse,gc,hashlib,json,random,time,traceback
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.style_control import mt5_input
from replycoach.generate.train_v5 import collate,epoch_batches,validation_loss
from replycoach.generate.v5_provider import DECODING
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/common/home/users/h/hm.teoh.2024/ReplyCoachSG_standard_val_v1/artifacts/full_lr3e4/best')
PUBLIC='data/public/sealion_dialogue_v1'
STYLES=('STANDARD','LOCAL')

def read(path):return json.loads((ROOT/path).read_text())
def rows(path):return [json.loads(l) for l in (ROOT/path).read_text().splitlines() if l.strip()]
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def task_rows(records):
    return [dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],style=s,response=r['replies']['standard' if s=='STANDARD' else 'local']) for r in records for s in STYLES]
def source(r,style=None):return mt5_input(style or r['style'],r['message'],r['context'],True,r.get('speakers'))
def encode(data,tok,public=False):
    result=[];excluded=[]
    for r in data:
        a=tok(source(r,'REPLY' if public else None))['input_ids'];b=tok(r['response'])['input_ids']
        if len(a)>(384 if public else 192) or len(b)>(128 if public else 48):
            if not public:raise ValueError('Task budget exceeded: '+r['item_id'])
            excluded.append(r['item_id']);continue
        result.append((a,b))
    if not result:raise ValueError('Empty training or validation group')
    return result,excluded

def train_pass(model,opt,data,tok,seed,epoch):
    model.train();opt.zero_grad(set_to_none=True);batches=epoch_batches(len(data),4,seed,epoch);total=0.
    for index,indices in enumerate(batches):
        ids,mask,labels=collate([data[i] for i in indices],tok.pad_token_id)
        loss=model(input_ids=ids.cuda(),attention_mask=mask.cuda(),labels=labels.cuda()).loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        group=min(2,len(batches)-(index//2)*2);(loss/group).backward();total+=float(loss.detach())
        if (index+1)%2==0 or index+1==len(batches):
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
            opt.step();opt.zero_grad(set_to_none=True)
    return total/len(batches)

def metrics(model,groups,tok):
    values={k:validation_loss(model,v,4,tok.pad_token_id,'cuda') for k,v in groups.items()}
    return dict(groups=values,selection_score=sum(values.values())/len(values))
def save(model,tok,path):
    model.save_pretrained(path);tok.save_pretrained(path)
    dump(path/'v5_input.json',dict(speaker_labels=True,joint=False,max_source=192,max_target=48))
@torch.no_grad()
def generate(model,tok,data):
    model.eval();model.config.use_cache=True;result=[]
    for r in data:
        case=dict(item_id=r['item_id'],styles={})
        for style in STYLES:
            inputs=tok(source(r,style),return_tensors='pt')
            if inputs['input_ids'].shape[-1]>192:raise ValueError('Generation source budget')
            output=model.generate(**{k:v.cuda() for k,v in inputs.items()},max_new_tokens=48,**DECODING)
            ids=output[0].tolist();text=tok.decode(output[0],skip_special_tokens=True)
            case['styles'][style]=dict(text=text,token_ids=ids,status='empty' if not text.strip() else 'complete' if tok.eos_token_id in ids else 'length_limit')
        result.append(case)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['public','control'],required=True);args=ap.parse_args()
    out=ROOT/'artifacts'/args.arm
    if (out/'results.json').exists():raise FileExistsError('Fresh output required')
    recipe=read('mt5_recipe.json')
    for p,h in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Frozen file mismatch: '+p)
    train=task_rows(rows('data/dev/train.jsonl'));val=task_rows(rows('data/dev/validation.jsonl'))
    if len(train)!=380 or len(val)!=100:raise ValueError('Wrong project counts')
    if {r['item_id'] for r in train}&{r['item_id'] for r in val}:raise ValueError('Task split overlap')
    public=rows(PUBLIC+'/train.jsonl');pv=rows(PUBLIC+'/validation.jsonl')
    if {r['group_id'] for r in public}&{r['group_id'] for r in pv}:raise ValueError('Public group overlap')
    torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(BASE)
    task,_=encode(train,tok);valid={s:encode([r for r in val if r['style']==s],tok)[0] for s in STYLES}
    pub,excluded=encode(public,tok,True);pubval={c:encode([r for r in pv if r['corpus']==c],tok,True)[0] for c in ['cosem','dailydialog']}
    fresh=read('data/evaluation/sealion_dialogue_v1/inputs.json');old=read('data/evaluation/sealion_dual_v1/inputs.json')
    report=dict(status='running',baseline_reused20=read('data/evaluation/mt5_dialogue_v1/reference.json'),arm=args.arm,recipe=recipe,gpu=torch.cuda.get_device_name(0),public_eligible=len(pub),public_budget_exclusions=excluded,runs={},default_changed=False)
    dump(out/'results.json',report)
    try:
        for seed in [425,426]:
            started=time.monotonic();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
            model=AutoModelForSeq2SeqLM.from_pretrained(BASE).cuda();model.gradient_checkpointing_enable();model.config.use_cache=False
            run=dict(seed=seed,public_history=[],task_history=[],outputs={});report['runs'][str(seed)]=run;folder=out/f's{seed}'
            run['base_validation']=metrics(model,valid,tok)
            run['outputs']['base_direct_new12']=generate(model,tok,fresh);model.config.use_cache=False
            dump(out/'results.json',report)
            initial=metrics(model,pubval,tok);run['public_history'].append(dict(examples_seen=0,**initial));run['public_selected_examples']=0
            if args.arm=='public':
                save(model,tok,folder/'public_best');best=initial['selection_score'];opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
                order=torch.randperm(len(pub),generator=torch.Generator().manual_seed(1425+seed-425)).tolist()
                for start in range(0,len(order),2000):
                    part=[pub[i] for i in order[start:start+2000]];loss=train_pass(model,opt,part,tok,seed,1+start//2000);scores=metrics(model,pubval,tok)
                    seen=min(start+2000,len(pub));run['public_history'].append(dict(examples_seen=seen,train_batch_mean_ce=loss,**scores))
                    if scores['selection_score']<best:best=scores['selection_score'];run['public_selected_examples']=seen;save(model,tok,folder/'public_best')
                    dump(out/'results.json',report);print(args.arm,seed,'public',seen,scores,flush=True)
                del opt,model;gc.collect();torch.cuda.empty_cache();model=AutoModelForSeq2SeqLM.from_pretrained(folder/'public_best').cuda();model.gradient_checkpointing_enable();model.config.use_cache=False
            # Identical task random state/order and optimizer policy in both arms.
            torch.manual_seed(seed+10000);torch.cuda.manual_seed_all(seed+10000)
            run['task_initial_validation']=metrics(model,valid,tok);best=None;stale=0;opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
            for epoch in range(1,9):
                loss=train_pass(model,opt,task,tok,seed,epoch);scores=metrics(model,valid,tok);run['task_history'].append(dict(epoch=epoch,train_batch_mean_ce=loss,**scores))
                if best is None or scores['selection_score']<best:best=scores['selection_score'];run['best_epoch']=epoch;stale=0;save(model,tok,folder/'best')
                else:stale+=1
                dump(out/'results.json',report);print(args.arm,seed,'task',epoch,scores,flush=True)
                if stale>=2:break
            del opt,model;gc.collect();torch.cuda.empty_cache();model=AutoModelForSeq2SeqLM.from_pretrained(folder/'best').cuda()
            run['selected_validation']=metrics(model,valid,tok)
            for name,data in [('new12',fresh),('reused20',old),('reused_validation',[r for r in val if r['style']=='STANDARD'])]:
                run['outputs'][name]=generate(model,tok,data);dump(out/'results.json',report)
            run['seconds']=time.monotonic()-started;del model;gc.collect();torch.cuda.empty_cache()
        report['status']='complete';dump(out/'results.json',report)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());dump(out/'results.json',report);raise
if __name__=='__main__':main()
