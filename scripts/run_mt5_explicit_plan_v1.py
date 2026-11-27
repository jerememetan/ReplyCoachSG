"""Learn source-only notes, then explicitly condition reply generation on them."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import traceback
import unittest

import torch
from peft import PeftModel
from transformers import AutoTokenizer
import run_mt5_adapter_switch_v1 as core
from replycoach.generate.mt5_adapter_switch_v1 import new_task_adapter
from replycoach.generate.mt5_matched_v1 import reply_loss,source
from replycoach.generate.mt5_explicit_plan_v1 import planner_source,planned_reply_source

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
TOK=None
R=None


def read(p):return json.loads((ROOT/p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,r):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n');temp.replace(p)
def clear():gc.collect();torch.cuda.empty_cache()
def seed(n):random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def digest(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if 'lora_' in n:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def encoded(text,target,max_source,max_target):
    x=TOK(text)['input_ids'];y=TOK(target)['input_ids']
    assert len(x)<=max_source and len(y)<=max_target and y[-1]==TOK.eos_token_id
    return x,y
def loss(model,batch):
    x,a,y=core.tensor(batch);v=reply_loss(model(input_ids=x,attention_mask=a,labels=y,use_cache=False).logits,y)
    assert torch.isfinite(v);return v


@torch.no_grad()
def token_ce(model,rows):return core.validation(model,rows)['token_ce']


@torch.no_grad()
def generate(model,texts,max_source,max_new):
    model.eval();out=[]
    for start in range(0,len(texts),4):
        batch=TOK(texts[start:start+4],padding=True,return_tensors='pt');assert batch['input_ids'].shape[1]<=max_source
        result=model.generate(**{k:v.cuda() for k,v in batch.items()},num_beams=4,no_repeat_ngram_size=3,
            max_new_tokens=max_new,do_sample=False,use_cache=True)
        for ids in result:
            text=TOK.decode(ids,skip_special_tokens=True)
            out.append(dict(text=text,status='failed_empty' if not text.strip() else 'complete' if TOK.eos_token_id in ids.tolist() else 'length_limit'))
    return out


def model_for_training(adapter=None):
    seed(R['seed'])
    base=core.load_base(ROOT/'model')
    model=new_task_adapter(base) if adapter is None else PeftModel.from_pretrained(base,adapter,is_trainable=True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==1769472
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    return model


def verify():
    for p,h in R['files_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in read('source_manifest.json')['files_sha256'].items():assert sha(ROOT/'model'/p)==h,p
    assert sha(ROOT/'protected_team.jsonl')==R['team_original_sha256']
    assert sha(Path(R['reply_initial_adapter'])/'adapter_model.safetensors')==R['reply_initial_sha256']
    assert 'L40S' in torch.cuda.get_device_name(0)
    tests=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mt5_explicit_plan_v1.py'))
    assert tests.wasSuccessful() and tests.testsRun==8


def planner():
    folder=OUT/'planner';assert not folder.exists();folder.mkdir(parents=True)
    train=read('data/planner_train.json');valid=read('data/planner_validation.json')
    enc={k:[encoded(planner_source(r),r['gold_notes'],256,160) for r in rows] for k,rows in train.items()}
    val={k:[encoded(planner_source(r),r['gold_notes'],256,160) for r in rows] for k,rows in valid.items()}
    save('preflight.json',dict(status='passed',tests_passed=8,source_model_and_payload_verified=True,
        planner_train=sum(len(v) for v in enc.values()),planner_validation=sum(len(v) for v in val.values()),
        no_truncation=True,gpu=torch.cuda.get_device_name(0)))
    model=model_for_training();params=[p for p in model.parameters() if p.requires_grad]
    frozen=[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
    opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=.01,foreach=False)
    report=dict(status='training',history=[],optimizer_steps=0,initial_digest=digest(model))
    best=float('inf')
    def snapshot(step):
        nonlocal best
        scores={k:token_ce(model,rows) for k,rows in val.items()};scores['macro_ce']=sum(scores.values())/len(scores)
        p=folder/f'step_{step:04d}';model.save_pretrained(p)
        report['history'].append(dict(step=step,validation=scores,sha256=sha(p/'adapter_model.safetensors')))
        if scores['macro_ce']<best:best=scores['macro_ce'];report['selected_step']=step
        save(folder/'training.json',report);print('Planner validation',step,scores,flush=True)
    def stream(count,offset):
        cycle=0
        while True:
            ids=list(range(count));random.Random(R['seed']+offset+cycle).shuffle(ids)
            yield from ids;cycle+=1
    creative,factual=stream(len(enc['creative']),0),stream(len(enc['factual']),10000)
    try:
        snapshot(0)
        for step in range(1,451):
            seed(R['seed']*10000+step);model.train();opt.zero_grad(set_to_none=True)
            for part,indices in [('creative',[next(creative) for _ in range(4)]),('factual',[next(factual) for _ in range(4)])]:
                for i in indices:(loss(model,[enc[part][i]])/8).backward()
            assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
            norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
            opt.step();opt.zero_grad(set_to_none=True);report['optimizer_steps']=step
            if step%25==0:save(folder/'training.json',report);print('Planner',step,'/450',flush=True)
            if step in R['planner']['snapshots'][1:]:snapshot(step)
        assert frozen==[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
        report.update(training_status='complete',frozen_base_unchanged=True)
        (folder/'selected').symlink_to(f"step_{report['selected_step']:04d}",target_is_directory=True)
        save(folder/'training.json',report)
        del model,params,opt;clear()
        model=PeftModel.from_pretrained(core.load_base(ROOT/'model'),folder/'selected',is_trainable=False).eval()
        # Predict once per unique conversation; IDs shared across validation/evaluation reuse exactly the same notes.
        all_rows=read('data/ordinary.json')+read('data/new_train.json')+read('data/ordinary_validation.json')+read('data/new_validation.json')+read('data/evaluation.json')
        unique={}
        for row in all_rows:
            if row['item_id'] in unique:assert planner_source(unique[row['item_id']])==planner_source(row)
            unique.setdefault(row['item_id'],row)
        rows=list(unique.values());texts=[planner_source(r) for r in rows]
        generated=generate(model,texts,256,160)
        cache={r['item_id']:dict(plan=g['text'],status=g['status'],planner_input_sha256=__import__('hashlib').sha256(t.encode()).hexdigest(),
            origin='freely_generated_selected_planner') for r,g,t in zip(rows,generated,texts,strict=True)}
        # Also check the actual predicted-note reply inputs before any responder trains.
        for row in rows:assert len(TOK(planned_reply_source(row,cache[row['item_id']]['plan']))['input_ids'])<=384
        save('predicted_notes.json',cache)
        report.update(status='complete',prediction_cache_cases=len(cache),
            prediction_flags={k:v['status'] for k,v in cache.items() if v['status']!='complete'},
            selected_sha256=sha(folder/'selected/adapter_model.safetensors'),cache_sha256=sha(ROOT/'predicted_notes.json'))
        save(folder/'training.json',report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc());save(folder/'training.json',report);raise


def active(model,name):model.set_adapter(name);model.requires_grad_(False);model.eval()


@torch.no_grad()
def paired_route(model,rows,texts,plans,route):
    assert len(rows)==len(texts)==len(plans)
    active(model,'default');english=generate(model,texts,384,48)
    ready=[i for i,e in enumerate(english) if e['text'].strip()]
    active(model,'rewriter')
    local=generate(model,[source(rows[i],'LOCAL_REWRITE',english[i]['text']) for i in ready],384,48)
    locals_=dict(zip(ready,local,strict=True));out=[]
    for i,(row,e,notes) in enumerate(zip(rows,english,plans,strict=True)):
        sg=locals_.get(i,dict(text='',status='blocked_empty_anchor'))
        out.append(dict(item_id=row['item_id'],scope=row['scope'],context=row['context'],speakers=row['speakers'],message=row['message'],
            route=route,notes=notes,reply_input_sha256=hashlib.sha256(texts[i].encode()).hexdigest(),
            english=e['text'],singlish=sg['text'],standard_anchor=e['text'],english_status=e['status'],singlish_status=sg['status']))
    return out


def evaluate(arm,checkpoint,label,cache):
    rows=read('data/evaluation.json')
    model=PeftModel.from_pretrained(core.load_base(ROOT/'model'),checkpoint,is_trainable=False)
    model.load_adapter(R['frozen_rewriter'],adapter_name='rewriter',is_trainable=False)
    routes={}
    if arm=='unchanged_baseline':
        routes['standard']=paired_route(model,rows,[source(r,'STANDARD') for r in rows],['']*len(rows),'unchanged_standard_baseline')
    else:
        actual=[cache[r['item_id']]['plan'] if arm=='explicit_notes' else '' for r in rows]
        name='predicted_notes' if arm=='explicit_notes' else 'no_notes_control'
        routes[name]=paired_route(model,rows,[planned_reply_source(r,n) for r,n in zip(rows,actual,strict=True)],actual,name)
        if arm=='explicit_notes':
            routes['same_model_no_notes']=paired_route(model,rows,[planned_reply_source(r,'') for r in rows],['']*len(rows),'same_model_no_notes')
            gold=[r for r in rows if r.get('gold_notes')]
            routes['oracle_notes_diagnostic']=paired_route(model,gold,[planned_reply_source(r,r['gold_notes']) for r in gold],
                [r['gold_notes'] for r in gold],'oracle_notes_diagnostic_only')
    result=dict(status='complete',arm=arm,checkpoint=str(checkpoint),routes=routes,
        actual_planner_notes={r['item_id']:cache[r['item_id']] for r in rows},
        oracle_never_used_in_normal_route=True,refusal_gate_used=False,frozen_rewriter=R['frozen_rewriter'],
        flags=[dict(route=k,item_id=r['item_id'],english_status=r['english_status'],singlish_status=r['singlish_status']) for k,rs in routes.items() for r in rs if r['english_status']!='complete' or r['singlish_status']!='complete'])
    save(OUT/arm/(label+'_evaluation.json'),result)
    del model;clear()


def responder(arm):
    assert read('outputs/planner/training.json')['status']=='complete'
    folder=OUT/arm;assert not folder.exists();folder.mkdir(parents=True)
    cache=read('predicted_notes.json')
    if arm=='no_notes_control':evaluate('unchanged_baseline',R['reply_initial_adapter'],'reference',cache)
    old,new=read('data/ordinary.json'),read('data/new_train.json')
    oldvalid,newvalid=read('data/ordinary_validation.json'),read('data/new_validation.json')
    def note(row,origin='predicted'):
        if arm=='no_notes_control':return ''
        return row['gold_notes'] if origin=='gold' else cache[row['item_id']]['plan']
    def enc(row,target,origin='predicted'):return encoded(planned_reply_source(row,note(row,origin)),target,384,48)
    oldenc=[enc(r,r['response']) for r in old]
    newenc={origin:[[enc(r,a['english'],origin) for a in r['alternatives']] for r in new] for origin in ['gold','predicted']}
    oldval=[enc(r,r['response']) for r in oldvalid]
    newval=[[enc(r,a['english']) for a in r['alternatives']] for r in newvalid]
    model=model_for_training(R['reply_initial_adapter']);params=[p for p in model.parameters() if p.requires_grad]
    frozen=[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
    opt=torch.optim.AdamW(params,lr=3e-5,weight_decay=.01,foreach=False)
    report=dict(status='training',arm=arm,history=[],optimizer_steps=0,initial_digest=digest(model),
        cache_sha256=sha(ROOT/'predicted_notes.json'),schedule_sha256=sha(ROOT/'schedule.json'))
    best=float('inf')
    def snapshot(step):
        nonlocal best
        model.eval();oldce=token_ce(model,oldval)
        with torch.no_grad():newce=sum(sum(float(loss(model,[e])) for e in refs)/3 for refs in newval)/len(newval)
        score=.5*(oldce+newce);path=folder/f'step_{step:04d}';model.save_pretrained(path)
        report['history'].append(dict(step=step,old_token_ce=oldce,new_mean_reference_ce=newce,selection_score=score,sha256=sha(path/'adapter_model.safetensors')))
        if score<best:best=score;report['selected_step']=step
        save(folder/'training.json',report);print(arm,'validation',step,score,flush=True)
    try:
        snapshot(0)
        for step,entry in enumerate(read('schedule.json'),1):
            seed(R['seed']*10000+step);model.train();opt.zero_grad(set_to_none=True)
            for i in entry['ordinary']:(loss(model,[oldenc[i]])/8).backward()
            for i,v,origin in zip(entry['new'],entry['refs'],entry['notes_origin'],strict=True):(loss(model,[newenc[origin][i][v]])/8).backward()
            assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
            norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
            opt.step();opt.zero_grad(set_to_none=True);report['optimizer_steps']=step
            if step%20==0:save(folder/'training.json',report);print(arm,step,'/300',flush=True)
            if step in R['responders']['snapshots'][1:]:snapshot(step)
        assert frozen==[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
        report.update(training_status='complete',frozen_base_unchanged=True)
        (folder/'selected').symlink_to(f"step_{report['selected_step']:04d}",target_is_directory=True)
        save(folder/'training.json',report)
        del model,params,opt;clear()
        evaluate(arm,folder/'selected','selected',cache)
        if report['selected_step']==300:report['final_reuses_selected']=True
        else:evaluate(arm,folder/'step_0300','final',cache)
        for p,h in read('source_manifest.json')['files_sha256'].items():assert sha(ROOT/'model'/p)==h
        assert sha(Path(R['reply_initial_adapter'])/'adapter_model.safetensors')==R['reply_initial_sha256']
        report.update(status='complete',source_base_and_initial_adapter_unchanged=True,default_changed=False)
        save(folder/'training.json',report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc());save(folder/'training.json',report);raise


def main():
    global TOK,R
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['planner','no_notes_control','explicit_notes'],required=True);args=parser.parse_args()
    assert Path.home().name=='hm.teoh.2024'
    torch.set_num_threads(4);R=read('recipe.json');TOK=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);core.TOK=TOK
    verify()
    if args.stage=='planner':planner()
    else:responder(args.stage)


if __name__=='__main__':main()
