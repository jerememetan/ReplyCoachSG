"""Personal-account style experiments; source bundle and English adapter read only."""
import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
BUNDLE=Path.home()/'ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1'
sys.path[:0]=[str(ROOT),str(BUNDLE/'src'),str(BUNDLE/'deps')]
import sealion_style_v1 as style
import torch
from peft import PeftModel
from replycoach.generate.sealion_v1 import SEALionCausalLM,config_for_checkpoint,checkpoint_tokenizer
from replycoach.generate.sealion_rewrite_v1 import encode_rewrite,rewrite_source,BOUNDARY
from replycoach.generate.sealion_dual_v1 import reply_loss
from replycoach.generate.sealion_beam_v1 import beam_decode


def read(name):return json.loads((ROOT/name).read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temporary.replace(path)


def load(checkpoint=None,trainable=False):
    model,info=SEALionCausalLM.from_pretrained(BUNDLE/'base',
        config=config_for_checkpoint(BUNDLE/'base'),dtype=torch.bfloat16,
        device_map={'':'cuda'},local_files_only=True,output_loading_info=True)
    problems={key:list(info.get(key) or []) for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']}
    assert not any(problems.values()),problems
    assert sum(p.numel() for p in model.parameters())==3178255360
    model=PeftModel.from_pretrained(model,checkpoint or BUNDLE/'adapters/singlish',is_trainable=trainable)
    if not trainable:model.requires_grad_(False)
    return model


def encoded_loss(model,encoded):
    data={key:torch.tensor([encoded[key]],device='cuda') for key in ['input_ids','attention_mask','labels']}
    labels=data.pop('labels')
    return reply_loss(model(**data,use_cache=False).logits,labels)


@torch.no_grad()
def score_text(model,tok,row,text):
    encoded=encode_rewrite(dict(row,response=text),tok,384)
    value=float(encoded_loss(model,encoded))
    if not torch.isfinite(torch.tensor(value)):raise FloatingPointError('Nonfinite candidate likelihood')
    return value


@torch.no_grad()
def decode(model,tok,row,beams=4):
    ids=tok.encode(rewrite_source(row,row['standard_anchor']),add_special_tokens=False)
    return beam_decode(model,tok,ids,BOUNDARY,num_beams=beams,
                       no_repeat_ngram_size=3,max_new_tokens=64,length_penalty=1.)


def choose(anchor,candidates,lexicon,copy_weight,style_weight):
    valid,scored=style.rank_candidates(anchor,candidates,lexicon,copy_weight,style_weight)
    if valid:
        item=valid[0]
        return dict(text=item['text'],provenance=item['provenance'],status='selected',
                    score=item['rerank_score'],guard=item['guard'],copy=item['copy'],
                    style_score=item['style_score']),scored
    # Surface guards can reject valid paraphrases. Preserve a clearly labelled
    # English fallback rather than making a factual change just to lower copying.
    return dict(text=anchor,provenance='english_fallback_no_guard_pass',status='fallback',
                guard=style.guard(anchor,anchor),copy=style.copy_features(anchor,anchor),
                style_score=style.style_score(anchor,lexicon)),scored


@torch.no_grad()
def candidates(model,tok,row,baseline):
    anchor=row['standard_anchor'];prompt=tok.encode(rewrite_source(row,anchor),add_special_tokens=False)
    raw=style.beam_nbest(model,tok,prompt,BOUNDARY,num_beams=8,returns=8)
    pool={}
    def add(text,provenance,status='complete'):
        if not text.strip():return
        key=style.canonical(text)
        if key not in pool:pool[key]=dict(text=text,provenance=provenance,status=status)
    for item in raw:add(item['text'],'model:nbest8',item['status'])
    add(baseline,'model:beam4')
    add(anchor,'english_fallback')
    model_keys=set(pool)
    for item in style.grammar_candidates(anchor):add(item['text'],item['provenance'])
    for item in pool.values():item['mean_nll']=score_text(model,tok,row,item['text'])
    model_pool=[item for key,item in pool.items() if key in model_keys]
    return model_pool,list(pool.values()),raw


def metrics(rows,method):
    scopes={}
    for scope in sorted({row['scope'] for row in rows}):
        group=[row for row in rows if row['scope']==scope]
        outputs=[(row,row['methods'][method]) for row in group if method in row['methods']]
        eligible=[(row,out) for row,out in outputs if style.copy_eligible(row['standard_anchor'])]
        scopes[scope]=dict(cases=len(outputs),eligible_copy_cases=len(eligible),
            exact_copies=sum(style.canonical(row['standard_anchor'])==style.canonical(out['text']) for row,out in outputs),
            unnecessary_copies=sum(style.copy_features(row['standard_anchor'],out['text'])['unnecessary_copy'] for row,out in eligible),
            particle_only_changes=sum(style.copy_features(row['standard_anchor'],out['text'])['particle_only_change'] for row,out in outputs),
            heuristic_guard_pass=sum(style.guard(row['standard_anchor'],out['text'])['passed'] for row,out in outputs),
            mean_style_heuristic=sum(style.style_score(out['text'],LEXICON) for _,out in outputs)/max(1,len(outputs)),
            fallbacks=sum(out.get('status')=='fallback' for _,out in outputs),
            failures=sum(out.get('status') not in {'complete','selected','fallback'} for _,out in outputs),
            warning='Guard passage and style score are heuristics, not human quality or correctness scores.')
    return scopes


def evaluate(model,tok,cases,output,report,expanded=True,repair_source=False):
    model.eval();rows=report.setdefault('outputs',[])
    for index,row in enumerate(cases):
        actual=decode(model,tok,row)
        baseline=actual['reply']
        if repair_source and row.get('baseline_local') is not None:
            if baseline!=row['baseline_local']:
                report.setdefault('repair25_replay_differences',[]).append(dict(item_id=row['item_id'],expected=row['baseline_local'],actual=baseline))
        result=dict(item_id=row['item_id'],scope=row['scope'],standard_anchor=row['standard_anchor'],
            context=row['context'],speakers=row['speakers'],message=row['message'],
            methods=dict(beam4=dict(text=baseline,provenance='model:beam4',status=actual['status'])))
        if expanded:
            model_pool,all_pool,raw=candidates(model,tok,row,baseline)
            result['candidate_pool']=all_pool
            result['native_nbest']=raw
            settings={'guarded_model':(model_pool,0.,0.),'copy_penalty':(model_pool,.5,0.),
                      'lexicon_guidance':(all_pool,0.,.2),'combined':(all_pool,.5,.2)}
            for method,(pool,copy_weight,style_weight) in settings.items():
                selected,_=choose(row['standard_anchor'],pool,LEXICON,copy_weight,style_weight)
                result['methods'][method]=selected
        rows.append(result)
        report['completed_cases']=index+1
        save(output,report)
        if (index+1)%10==0:print('Evaluation',output.parent.name,index+1,'/',len(cases),flush=True)
    methods=sorted({method for row in rows for method in row['methods']})
    report['metrics']={method:metrics(rows,method) for method in methods}
    report['status']='complete'
    save(output,report)


@torch.no_grad()
def validation(model,replay,pairs):
    model.eval();summed=0.;tokens=0
    for item in replay:
        value=float(encoded_loss(model,item));count=sum(label!=-100 for label in item['labels'][1:])
        summed+=value*count;tokens+=count
    pos=[];copies=[]
    for a,b in pairs:
        pos.append(float(encoded_loss(model,a)));copies.append(float(encoded_loss(model,b)))
    ordinary=summed/tokens
    positive=sum(pos)/len(pos);copied=sum(copies)/len(copies)
    margin=sum(max(0.,.2+a-b) for a,b in zip(pos,copies,strict=True))/len(pos)
    return dict(replay_token_ce=ordinary,preferred_example_mean_ce=positive,
        copy_example_mean_ce=copied,copy_ranking_win_fraction=sum(a<b for a,b in zip(pos,copies,strict=True))/len(pos),
        margin_violation=margin,selection_score=.8*ordinary+.2*positive+.05*margin,
        scope='Teacher-forced validation on42 existing pairs and11 copy contrasts; not generated reply accuracy.')


def adapter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if 'lora_' in name:
            h.update(name.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def clear():gc.collect();torch.cuda.empty_cache()


def train_arm(arm,tok,train,valid,auxiliary,aux_valid,folder,recipe):
    torch.manual_seed(425);torch.cuda.manual_seed_all(425)
    model=load(trainable=True)
    model.gradient_checkpointing_enable();model.enable_input_require_grads();model.config.use_cache=False
    params=[p for name,p in model.named_parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==10485760
    assert all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)
    frozen=[(name,id(p),p._version) for name,p in model.named_parameters() if 'lora_' not in name]
    count=recipe['training']['optimizer_steps'];rng=random.Random(425)
    order=[]
    while len(order)<count*8:
        part=list(range(len(train)));rng.shuffle(part);order.extend(part)
    aux_order=[]
    while len(aux_order)<count:
        part=list(range(len(auxiliary)));rng.shuffle(part);aux_order.extend(part)
    schedule=dict(replay=order[:count*8],auxiliary=aux_order[:count])
    initial=validation(model,valid,aux_valid)
    report=dict(status='training',arm=arm,history=[dict(step=0,scores=initial)],
        initial_adapter_digest=adapter_digest(model),trainable_parameters=sum(p.numel() for p in params),
        schedule_sha256=hashlib.sha256(json.dumps(schedule,sort_keys=True).encode()).hexdigest(),
        initial_validation=initial,selected_step=0,optimizer_steps=0,copy_margin_active_steps=0,
        loss='mean8 replay CE +0.1 preferred CE +weight*relu(0.2+preferredCE-copyCE)',
        copy_margin_weight=0. if arm=='control' else .1,default_changed=False)
    output=folder/'training.json';folder.mkdir(parents=True,exist_ok=True)
    selected=folder/'selected';model.save_pretrained(selected)
    best=initial['selection_score'];initial_ce=initial['replay_token_ce'];save(output,report)
    optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01,foreach=False)
    optimizer.zero_grad(set_to_none=True)
    for index in range(count):
        torch.manual_seed(425000+index);torch.cuda.manual_seed_all(425000+index);model.train()
        for which in schedule['replay'][index*8:(index+1)*8]:
            loss=encoded_loss(model,train[which])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite replay loss')
            (loss/8).backward()
        a,b=auxiliary[schedule['auxiliary'][index]]
        positive=encoded_loss(model,a);copy=encoded_loss(model,b)
        auxiliary_loss=style.copy_margin_loss(positive,copy,report['copy_margin_weight'])
        if not torch.isfinite(auxiliary_loss):raise FloatingPointError('Nonfinite copy auxiliary loss')
        auxiliary_loss.backward()
        if .2+float(positive.detach())-float(copy.detach())>0:report['copy_margin_active_steps']+=1
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite adapter gradient')
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        report['optimizer_steps']=index+1
        if (index+1)%5==0:
            report['latest_losses']=dict(preferred=float(positive.detach()),copy=float(copy.detach()),grad_norm=float(norm))
            save(output,report);print('Training',arm,index+1,'/',count,report['latest_losses'],flush=True)
        if index+1 in [25,50,count]:
            scores=validation(model,valid,aux_valid)
            eligible=scores['replay_token_ce']<=initial_ce*1.02
            chosen=eligible and scores['selection_score']<best
            model.save_pretrained(folder/f'step_{index+1:04d}')
            if chosen:
                best=scores['selection_score'];report['selected_step']=index+1;model.save_pretrained(selected)
            report['history'].append(dict(step=index+1,scores=scores,replay_guard_passed=eligible,selected=chosen))
            save(output,report)
        del positive,copy,auxiliary_loss,loss
    assert frozen==[(name,id(p),p._version) for name,p in model.named_parameters() if 'lora_' not in name]
    report.update(status='trained',ordinary_exposures=count*8,auxiliary_pair_exposures=count,
        selected_ordinary_exposures=report['selected_step']*8,selected_auxiliary_pair_exposures=report['selected_step'],
        frozen_base_unchanged=True,final_adapter_digest=adapter_digest(model),
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    assert report['final_adapter_digest']!=report['initial_adapter_digest']
    save(output,report)
    del optimizer,params,model;clear()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['algorithms','training'],required=True)
    args=parser.parse_args()
    assert Path.home().name=='hm.teoh.2024','All experiment jobs must use the personal account'
    recipe=read('recipe.json')
    for name,expected in recipe['files_sha256'].items():assert sha(ROOT/name)==expected,name
    assert read('preflight.json')['status']=='passed'
    torch.set_num_threads(4);torch.manual_seed(425);torch.cuda.manual_seed_all(425)
    assert 'L40S' in torch.cuda.get_device_name(0)
    spec=importlib.util.spec_from_file_location('export_loader',BUNDLE/'infer.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    manifest=loader.verify_files(BUNDLE)
    assert sha(BUNDLE/'manifest.json')==read('preflight.json')['source_manifest_sha256']
    weights={name:(BUNDLE/name).stat() for name in manifest['files'] if name.endswith('.safetensors')}
    tokenizer=checkpoint_tokenizer(BUNDLE/'base')
    train=read('data/train.json');valid=read('data/validation.json')
    cases=read('data/evaluation.json')+read('data/controls.json')
    global LEXICON
    LEXICON=style.fit_lexicon(train)
    folder=ROOT/'outputs'/args.mode
    assert not folder.exists(),'Fresh output directory required'
    folder.mkdir(parents=True)
    save(folder/'style_lexicon.json',LEXICON)
    started=time.monotonic()
    report=dict(status='running',mode=args.mode,recipe=recipe,gpu=torch.cuda.get_device_name(0),
        source_model='Exported SEA-LION repair25; English generator unchanged',
        data=read('data/data_manifest.json'),default_changed=False,quality_review='Pending; no human quality scores.')
    save(folder/'progress.json',report)
    try:
        if args.mode=='algorithms':
            model=load().eval();report['training_performed']=False
            evaluate(model,tokenizer,cases,folder/'results.json',report,repair_source=True)
            assert not report.get('repair25_replay_differences'),report.get('repair25_replay_differences')
            del model;clear()
        else:
            replay=[encode_rewrite(row,tokenizer,384) for row in train]
            validation_rows=[encode_rewrite(row,tokenizer,384) for row in valid]
            def pairs(rows):return [(encode_rewrite(row,tokenizer,384),
                encode_rewrite(dict(row,response=row['standard_anchor']),tokenizer,384)) for row in rows]
            auxiliary=pairs(read('data/copy_train.json'));aux_valid=pairs(read('data/copy_validation.json'))
            report['arms']={}
            for arm in ['control','copy_loss']:
                report['arms'][arm]=train_arm(arm,tokenizer,replay,validation_rows,auxiliary,aux_valid,folder/arm,recipe)
                save(folder/'progress.json',report)
            assert report['arms']['control']['initial_adapter_digest']==report['arms']['copy_loss']['initial_adapter_digest']
            assert report['arms']['control']['schedule_sha256']==report['arms']['copy_loss']['schedule_sha256']
            for arm in ['control','copy_loss']:
                model=load(folder/arm/'selected').eval()
                phase=dict(status='running',arm=arm,selected_step=report['arms'][arm]['selected_step'])
                evaluate(model,tokenizer,cases,folder/arm/'evaluation.json',phase,expanded=True)
                report['arms'][arm]['evaluation_metrics']=phase['metrics']
                del model;clear();save(folder/'progress.json',report)
            report['status']='complete';report['training_performed']=True
            save(folder/'results.json',report)
        for name,info in weights.items():
            current=(BUNDLE/name).stat()
            assert current.st_size==info.st_size and current.st_mtime_ns==info.st_mtime_ns
            if 'adapters/' in name:assert sha(BUNDLE/name)==manifest['files'][name]['sha256']
        report.update(status='complete',source_bundle_weights_unchanged=True,seconds=time.monotonic()-started)
        save(folder/'results.json',report);save(folder/'progress.json',report)
        print(json.dumps(dict(status='complete',mode=args.mode,seconds=report['seconds'],source_unchanged=True)),flush=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        save(folder/'failure.json',report)
        raise


if __name__=='__main__':main()
