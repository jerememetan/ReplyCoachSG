"""Four matched mT5 task mixtures; selected and final replies stay separate."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import traceback
import unittest

import torch
import transformers
import peft
from peft import PeftModel
from transformers import AutoTokenizer

import run_mt5_adapter_switch_v1 as core
from replycoach.generate.mt5_adapter_switch_v1 import new_task_adapter
from replycoach.generate.context_pair_ablation_v1 import encode_row
from replycoach.generate.mt5_matched_v1 import source, reply_loss
from replycoach.generate.mt5_cocktail_v1 import ARMS, audit_case, auxiliary_source

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
TOK=None
RECIPE=None
PRIMARY=None
AUX=None
VALID=None


def read(p):return json.loads((ROOT/p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,value):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def clear():gc.collect();torch.cuda.empty_cache()
def digest(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if 'lora_' in n:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def aux_encode(row,task):
    return TOK(auxiliary_source(row,task))['input_ids'],TOK(row['fact_target' if task=='FACTS' else 'answer_target'])['input_ids']
def loss(model,encoded):
    x,a,y=core.tensor(encoded)
    result=reply_loss(model(input_ids=x,attention_mask=a,labels=y,use_cache=False).logits,y)
    assert torch.isfinite(result)
    return result


@torch.no_grad()
def validation(model):
    scores={name:core.validation(model,rows) for name,rows in VALID.items()}
    scores['macro_ce']=(scores['english']['token_ce']+scores['rewrite']['token_ce'])/2
    return scores


@torch.no_grad()
def texts(model,sources,max_new_tokens=48):
    model.eval();out=[]
    for start in range(0,len(sources),4):
        batch=TOK(sources[start:start+4],padding=True,return_tensors='pt')
        assert batch['input_ids'].shape[1]<=192
        result=model.generate(**{k:v.cuda() for k,v in batch.items()},max_new_tokens=max_new_tokens,
            num_beams=4,no_repeat_ngram_size=3,do_sample=False,use_cache=True)
        for ids in result:
            text=TOK.decode(ids,skip_special_tokens=True)
            out.append(dict(text=text,status='failed_empty' if not text.strip() else 'complete' if TOK.eos_token_id in ids.tolist() else 'length_limit'))
    return out


def auxiliary_evaluation(model,rows):
    outputs=[]
    facts=texts(model,[auxiliary_source(r,'FACTS') for r in rows])
    answers=texts(model,[auxiliary_source(r,'ANSWERABILITY') for r in rows],max_new_tokens=12)
    norm=lambda x:'|'.join(p.strip().casefold() for p in x.split('|'))
    for row,f,a in zip(rows,facts,answers,strict=True):
        outputs.append(dict(item_id=row['item_id'],group_id=row['group_id'],state=row['state'],
            fact_target=row['fact_target'],answer_target=row['answer_target'],fact_output=f['text'],answer_output=a['text'],
            fact_correct=norm(f['text'])==norm(row['fact_target']),answer_correct=a['text'].strip().upper()==row['answer_target'],
            fact_status=f['status'],answer_status=a['status']))
    positive=[r for r in outputs if r['answer_target']=='ANSWERABLE'];negative=[r for r in outputs if r['answer_target']=='MISSING']
    return dict(outputs=outputs,metrics=dict(cases=len(outputs),fact_exact=sum(r['fact_correct'] for r in outputs),
        answerability_exact=sum(r['answer_correct'] for r in outputs),answerable_cases=len(positive),missing_cases=len(negative),
        false_missing=sum(r['answer_output'].strip().upper()=='MISSING' for r in positive),
        false_answerable=sum(r['answer_output'].strip().upper()=='ANSWERABLE' for r in negative),
        invalid_labels=sum(r['answer_output'].strip().upper() not in {'ANSWERABLE','MISSING'} for r in outputs),
        note='Auxiliary task exact matches, not reply quality. Labels are generated freely; no forced candidate choice.'))


def evaluate(model):
    new=read('data/challenge.json');cases=new+read('data/natural_challenge.json')+read('data/legacy.json')+read('data/diagnostics.json')
    english=texts(model,[source(r,'STANDARD') for r in cases])
    ready=[i for i,r in enumerate(english) if r['text'].strip()]
    local=texts(model,[source(cases[i],'LOCAL_REWRITE',english[i]['text']) for i in ready])
    by_index=dict(zip(ready,local,strict=True));outputs=[]
    for i,(row,e) in enumerate(zip(cases,english,strict=True)):
        sg=by_index.get(i,dict(text='',status='blocked_empty_anchor'))
        outputs.append(dict(item_id=row['item_id'],scope=row['scope'],context=row['context'],speakers=row['speakers'],message=row['message'],
            english=e['text'],singlish=sg['text'],standard_anchor=e['text'],english_status=e['status'],singlish_status=sg['status'],
            reply_gate_used=False,reference_conditioning_used=False))
    return dict(status='complete',outputs=outputs,
        auxiliary_challenge=auxiliary_evaluation(model,new),
        auxiliary_validation=auxiliary_evaluation(model,read('data/validation.json')),
        flags=[r['item_id'] for r in outputs if r['english_status']!='complete' or r['singlish_status']!='complete'],
        note='English and its own Singlish rewrite are freely generated. Auxiliary predictions do not gate or condition replies. Human semantic/style review pending.')


def preflight():
    for p,expected in RECIPE['files_sha256'].items():assert sha(ROOT/p)==expected,p
    manifest=read('source_manifest.json')
    for p,expected in manifest['files_sha256'].items():assert sha(ROOT/'model'/p)==expected,p
    assert sha(ROOT/'data/dev/train.jsonl')==RECIPE['team_original_sha256']
    assert (ROOT/'data/training_expansion/generated600_repair_v1/combined_train.jsonl').read_bytes().startswith((ROOT/'data/dev/train.jsonl').read_bytes())
    tests=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mt5_cocktail_v1.py'))
    assert tests.wasSuccessful() and tests.testsRun==8
    all_encoded=PRIMARY+sum(VALID.values(),[])
    for split in ['train','validation','challenge']:
        rows=read('data/'+split+'.json')
        for r in rows:
            audit_case(r)
            all_encoded.extend([aux_encode(r,'FACTS'),aux_encode(r,'ANSWERABILITY')])
    assert all(len(x)<=192 and len(y)<=48 and y[-1]==TOK.eos_token_id for x,y in all_encoded)
    assert len(PRIMARY)==1535 and len(AUX['FACTS'])==len(AUX['ANSWERABILITY'])==800
    assert 'L40S' in torch.cuda.get_device_name(0)
    result=dict(status='passed',tests_passed=8,encoded_examples=len(all_encoded),no_truncation=True,
        source_manifest_verified=True,team_data_unchanged=True,gpu=torch.cuda.get_device_name(0),
        versions=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__))
    return result


def initialize():
    seed=RECIPE['seed'];random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=new_task_adapter(core.load_base(ROOT/'model'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==1769472
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    return model


def gradient_smoke(result):
    model=initialize();model.eval()
    params=[p for p in model.parameters() if p.requires_grad]
    loss(model,PRIMARY[:2]).backward()
    gradients=lambda:[torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in params]
    before=gradients();interventions={}
    for task in ['FACTS','ANSWERABILITY']:
        before=gradients()
        (loss(model,AUX[task][:2])*0).backward()
        assert all(torch.equal(a,b) for a,b in zip(before,gradients(),strict=True))
        (loss(model,AUX[task][:2])/6).backward()
        assert any(not torch.equal(a,b) for a,b in zip(before,gradients(),strict=True))
        interventions[task]=dict(zero_weight_preserved_gradients=True,nonzero_weight_changed_gradients=True)
    assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
    smoke_rows=read('data/challenge.json')[:1]
    smoke=texts(model,[source(smoke_rows[0],'STANDARD')])
    result.update(actual_gradient_interventions=interventions,strict_model_load_passed=True,
        trainable_parameters=1769472,smoke_generation=smoke,optimizer_updates_performed=0)
    save('preflight.json',result)
    del model,params,before;clear()


def train(arm):
    folder=OUT/arm;report_path=folder/'report.json'
    if report_path.exists():raise FileExistsError(report_path)
    model=initialize();params=[p for p in model.parameters() if p.requires_grad]
    frozen=[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
    report=dict(status='training',arm=arm,seed=RECIPE['seed'],initial_adapter_digest=digest(model),history=[],
        recipe_sha256=sha(ROOT/'recipe.json'),schedule_sha256=sha(ROOT/'schedule.json'),
        weights=dict(zip(['facts','answerability'],ARMS[arm],strict=True)),preflight=preflight_result,
        default_changed=False,locked_test_read=False)
    optimizer=torch.optim.AdamW(params,lr=3e-4,weight_decay=.01,foreach=False)
    best=float('inf')
    def snapshot(step):
        nonlocal best
        scores=validation(model);p=folder/f'step_{step:04d}';model.save_pretrained(p)
        (p/'v5_input.json').write_text(json.dumps(dict(speaker_labels=True,joint=False,rewrite_local=True,max_source=192,max_target=48)))
        h=dict(step=step,validation=scores,checkpoint=str(p),sha256=sha(p/'adapter_model.safetensors'))
        report['history'].append(h)
        if scores['macro_ce']<best:
            best=scores['macro_ce'];report.update(selected_step=step,selected_checkpoint=str(p))
        save(report_path,report);print('Snapshot',arm,step,'macroCE',scores['macro_ce'],flush=True)
    try:
        snapshot(0)
        for step,entry in enumerate(read('schedule.json'),1):
            torch.manual_seed(RECIPE['seed']*1000000+step);torch.cuda.manual_seed_all(RECIPE['seed']*1000000+step)
            model.train();optimizer.zero_grad(set_to_none=True)
            main=[PRIMARY[i] for i in entry['main']];n=sum(len(y) for x,y in main)
            for i in range(0,12,3):
                part=main[i:i+3];v=loss(model,part);(v*sum(len(y) for x,y in part)/n).backward()
            for task,weight in zip(['FACTS','ANSWERABILITY'],ARMS[arm],strict=True):
                v=loss(model,[AUX[task][i] for i in entry['facts' if task=='FACTS' else 'answerability']]);(v*weight).backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            report.update(optimizer_steps=step,primary_exposures=step*12)
            if step%20==0:
                save(report_path,report);print('Training',arm,step,'/784',flush=True)
            if step in RECIPE['snapshots'][1:]:snapshot(step)
        assert frozen==[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
        report.update(training_status='complete',frozen_base_unchanged=True,final_adapter_digest=digest(model))
        save(report_path,report)
        del model,params,optimizer,v;clear()
        for label,step in [('selected',report['selected_step']),('final',784)]:
            if label=='final' and step==report['selected_step']:
                report['final_reuses_selected']=True;continue
            model=PeftModel.from_pretrained(core.load_base(ROOT/'model'),folder/f'step_{step:04d}',is_trainable=False).eval()
            evaluation=evaluate(model);save(folder/(label+'_evaluation.json'),evaluation)
            del model;clear()
        for p,h in read('source_manifest.json')['files_sha256'].items():assert sha(ROOT/'model'/p)==h,p
        report.update(status='complete',source_base_manifest_unchanged=True)
        save(report_path,report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc());save(report_path,report);raise


def main():
    global TOK,RECIPE,PRIMARY,AUX,VALID,preflight_result
    parser=argparse.ArgumentParser();parser.add_argument('--arms',nargs='+',choices=list(ARMS));parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    assert Path.home().name=='hm.teoh.2024'
    torch.set_num_threads(4);RECIPE=read('recipe.json')
    TOK=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);core.TOK=TOK
    PRIMARY=[encode_row(r,TOK) for r in read('data/primary.json')]
    aux=read('data/train.json');AUX={t:[aux_encode(r,t) for r in aux] for t in ['FACTS','ANSWERABILITY']}
    val=read('data/project_validation.json');VALID={name:[encode_row(r,TOK) for r in val if r['style']==tag] for name,tag in [('english','STANDARD'),('rewrite','LOCAL_REWRITE')]}
    preflight_result=preflight()
    if args.preflight_only:
        if (ROOT/'preflight.json').exists():raise FileExistsError('Preflight already recorded')
        gradient_smoke(preflight_result);print('PREFLIGHT PASSED',flush=True);return
    assert read('preflight.json')['strict_model_load_passed']
    assert args.arms
    for arm in args.arms:train(arm)


if __name__=='__main__':main()
