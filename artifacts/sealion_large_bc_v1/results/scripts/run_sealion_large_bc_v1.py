"""Matched B/C full-budget training; derived from verified audited restart."""
import argparse
import math
from replycoach.generate.sealion_large_bc_v1 import lr_factor
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import unittest

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from replycoach.generate import sealion_audited_restart_v1 as helper
from replycoach.generate.sealion_data import encode_reply,source
from replycoach.generate.sealion_rewrite_v1 import encode_rewrite,rewrite_source,BOUNDARY as REWRITE_BOUNDARY
from replycoach.generate.sealion_dual_v1 import reply_loss,BOUNDARY
from replycoach.generate.sealion_beam_v1 import beam_decode
from replycoach.generate.sealion_v1 import SEALionCausalLM,config_for_checkpoint,checkpoint_loading_problems,checkpoint_tokenizer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
R=None
TOK=None

def read(p):return json.loads((ROOT/p).read_text())
def save(p,value):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()
def seed(n):random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def clear():gc.collect();torch.cuda.empty_cache()
def verify():
    assert Path.home().name=='hm.teoh.2024'
    for p,h in R['payload_manifest'].items():assert sha(ROOT/p)==h,p
    for p,h in R['source_protected_hashes'].items():assert sha(ROOT/p)==h,p
    assert sha(Path(R['base_path'])/'model.safetensors')==R['base_weight_sha256']
    for name,h in R['public_hashes'].items():assert sha(Path(R['public_adapter'])/name)==h,name
    b=Path(R['original_bundle']);assert sha(b/'manifest.json')==R['original_manifest_sha256']
    manifest=json.loads((b/'manifest.json').read_text())
    for name in ['adapters/english/adapter_model.safetensors','adapters/singlish/adapter_model.safetensors']:
        assert sha(b/name)==manifest['files'][name]['sha256']
    assert any(x in torch.cuda.get_device_name(0).lower() for x in ['l40s','blackwell']), torch.cuda.get_device_name(0)

def base_model():
    path=Path(R['base_path'])
    model,info=SEALionCausalLM.from_pretrained(path,config=config_for_checkpoint(path),dtype=torch.bfloat16,
        device_map={'':0},local_files_only=True,output_loading_info=True)
    problems=checkpoint_loading_problems(info)
    assert not any(problems.values()) and not any(p.is_meta for p in model.parameters()),problems
    return model

def load(adapter,trainable=False):
    seed(R['seed']);model=PeftModel.from_pretrained(base_model(),adapter,is_trainable=trainable)
    expected=load_file(str(Path(adapter)/'adapter_model.safetensors'))
    actual=get_peft_model_state_dict(model)
    assert set(expected)==set(actual) and all(torch.equal(actual[k].detach().cpu(),v) for k,v in expected.items()),'Initial adapter did not reproduce saved weights'
    del expected,actual
    if trainable:
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert sum(p.numel() for _,p in named)==R['common']['trainable_parameters']
        assert all('lora_' in n for n,_ in named)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads();model.config.use_cache=False
    return model

def encode(row,stage,style=None):
    if stage=='rewrite':return encode_rewrite(dict(row,standard_anchor=row['standard'],response=row['local']),TOK,384)
    return encode_reply(dict(row,style=style,response=row[style.lower()]),TOK,384)

def tensors(e):return {k:torch.tensor([e[k]],device='cuda') for k in ['input_ids','attention_mask','labels']}
def loss(model,e):
    t=tensors(e);labels=t.pop('labels');out=model(**t,use_cache=False);v=reply_loss(out.logits,labels)
    assert torch.isfinite(v);return v

@torch.no_grad()
def ce(model,rows):
    model.eval();total=0.;tokens=0
    for e in rows:
        n=sum(y!=-100 for y in e['labels'][1:]);assert n>0;total+=float(loss(model,e))*n;tokens+=n
    return dict(token_ce=total/tokens,examples=len(rows),tokens=tokens)

def validation(model,stage,groups):
    values={k:ce(model,v) for k,v in groups.items()}
    if stage=='task':old=sum(values['old_'+s]['token_ce'] for s in ['STANDARD','LOCAL'])/2;new=sum(values['new_'+s]['token_ce'] for s in ['STANDARD','LOCAL'])/2
    else:old=values['old_rewrite']['token_ce'];new=values['new_rewrite']['token_ce']
    return dict(groups=values,old_macro_ce=old,new_macro_ce=new,selection_score=helper.selection(old,new))

@torch.no_grad()
def decode(model,row,style,anchor=None):
    text=helper.prompt(row,style,anchor)
    if style=='LOCAL_REWRITE':assert text==rewrite_source(row,anchor)
    else:assert text==source(row,style)
    ids=TOK.encode(text,add_special_tokens=False)
    if len(ids)+64>384:return dict(reply='',status='blocked_source_budget',prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
    out=beam_decode(model,TOK,ids,REWRITE_BOUNDARY if style=='LOCAL_REWRITE' else BOUNDARY,
        num_beams=4,max_new_tokens=64,length_penalty=1.,no_repeat_ngram_size=3)
    return dict(reply=out['reply'],status=out['status'],termination=out['termination'],prompt_sha256=hashlib.sha256(text.encode()).hexdigest())

@torch.no_grad()
def fixed_rewrites(model,rows):
    result=[]
    for r in rows:
        o=decode(model,r,'LOCAL_REWRITE',r['standard']);result.append(dict(item_id=r['item_id'],english=r['standard'],singlish=o['reply'],standard_anchor=r['standard'],status=o['status'],prompt_sha256=o['prompt_sha256'],anchor_provenance='Assistant-authored diagnostic reference; not the English generator output'))
    return result

def hardware():
    tests=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_sealion_*v1.py'))
    assert tests.wasSuccessful() and tests.testsRun==12
    row=read('data/task_train.json')[0]
    model=load(Path(R['public_adapter']),True)
    value=loss(model,encode(row,'task','STANDARD'));value.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    assert any(p.grad is not None and bool(p.grad.abs().max()>0) for p in params)
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
    assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
    save('hardware.json',dict(status='passed',gpu=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),torch=torch.__version__,cuda=torch.version.cuda,tests=tests.testsRun,forward_backward_finite=True,initial_weights_exactly_verified=True))
    del model,params,value;clear()

def data_verify():
    manifest=read('frozen_data/manifest.json');assert manifest['status']=='complete'
    for name,h in manifest['hashes'].items():assert sha(ROOT/'frozen_data'/name)==h,name
    expected={'train':R['data']['total_task_conversations'],
              'dev':R['data']['new_dev_conversations'],
              'evaluation':R['data']['new_final_conversations']}
    assert manifest['counts']==expected
    if 'accepted_snapshot' in R['data']:
        assert manifest['accepted_snapshot']==R['data']['accepted_snapshot']
        assert sha(ROOT/R['data']['accepted_snapshot']['path'])==R['data']['accepted_snapshot']['sha256']
    assert read('authoring/assistant_sample_review.json')['proceed'] is True

def preflight():
    # Apply explicit assistant-audited derivative corrections before the final
    # shared manifest is verified or either student arm can start.
    overlay_path=ROOT/'data/assistant_target_overlays.json'
    if overlay_path.exists():
        manifest=read('frozen_data/manifest.json');overlay_hash=sha(overlay_path)
        if manifest.get('assistant_overlay_sha256')!=overlay_hash:
            overlays=read('data/assistant_target_overlays.json')['corrections'];applied=[]
            original_hashes=dict(manifest['hashes'])
            for name,h in original_hashes.items():
                assert sha(ROOT/'frozen_data'/name)==h,name
                rows=read('frozen_data/'+name);changed=False
                for row in rows:
                    for correction in overlays:
                        if row['item_id'].split('-D')[0]!=correction['item_id'] or row['standard']!=correction['standard']:continue
                        source_hash=hashlib.sha256(json.dumps({k:row[k] for k in ['context','speakers','message']},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                        assert source_hash==correction['source_sha256'] and row['local']==correction['previous_local']
                        row['local']=correction['corrected_local'];changed=True
                        applied.append(dict(file=name,item_id=row['item_id'],note=correction['note']))
                if changed:save('frozen_data/'+name,rows)
            assert any(r['file']=='train_rewrite.json' for r in applied),'Audited correction did not match its intended pair'
            manifest.update(pre_overlay_hashes=original_hashes,assistant_overlay_sha256=overlay_hash,assistant_corrections=applied,
                hashes={name:sha(ROOT/'frozen_data'/name) for name in original_hashes})
            save('frozen_data/manifest.json',manifest)
    data_verify()
    trainrows=read('frozen_data/train_task.json');rw=read('frozen_data/train_rewrite.json')
    assert len(trainrows)==R['data']['total_task_conversations'] and len(rw)==R['data']['total_rewrite_pairs']
    helper.check_splits({s:read('frozen_data/'+s+'_task.json') for s in ['train','dev','evaluation']})
    maximum=0;count=0
    for stage,rows in [('task',trainrows),('rewrite',rw),('task',read('frozen_data/dev_task.json')),('rewrite',read('frozen_data/dev_rewrite.json'))]:
        for row in rows:
            for style in (['STANDARD','LOCAL'] if stage=='task' else ['LOCAL_REWRITE']):
                e=encode(row,stage,style);n=e['prompt_tokens'];prompt=helper.prompt(row,style,row['standard'] if stage=='rewrite' else None)
                assert e['input_ids'][:n]==TOK.encode(prompt,add_special_tokens=False)
                assert e['labels'][:n]==[-100]*n and e['labels'][n:]==e['input_ids'][n:] and e['labels'][-1]==TOK.eos_token_id
                assert len(TOK.encode(prompt,add_special_tokens=False))+64<=384
                maximum=max(maximum,len(e['input_ids']));count+=1
    save('preflight.json',dict(status='passed',encoded_targets=count,max_tokens=maximum,gpu=torch.cuda.get_device_name(0),data_manifest_sha256=sha(ROOT/'frozen_data/manifest.json'),no_truncation=True))
    # The saved original runs once on the same frozen final conversations.
    rows=read('frozen_data/evaluation_all.json')
    save('baseline_inputs.json',[{k:r[k] for k in ['item_id','context','speakers','message']} for r in rows])
    target=ROOT/'outputs/original/baseline.json';target.parent.mkdir(parents=True,exist_ok=True)
    with (target.parent/'inference.log').open('w') as log:
        result=subprocess.run([sys.executable,'-I',str(Path(R['original_bundle'])/'infer.py'),'--device','cuda','--inputs',str(ROOT/'baseline_inputs.json'),'--output',str(target)],stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode in [0,2] and target.exists()
    baseline=read(target);assert baseline['status']=='complete' and baseline['input_sha256']==sha(ROOT/'baseline_inputs.json')
    assert [r['item_id'] for r in baseline['outputs']]==[r['item_id'] for r in rows]
    save('ready.json',dict(status='ready',data_manifest_sha256=sha(ROOT/'frozen_data/manifest.json'),original_baseline_sha256=sha(target)))

def train(stage):
    assert read('ready.json')['status']=='ready';data_verify()
    folder=OUT/stage;assert not folder.exists();folder.mkdir(parents=True)
    if stage=='task':
        initial=Path(R['public_adapter']);rows=read('frozen_data/train_task.json')
        encoded=[encode(r,'task',s) for r in rows for s in ['STANDARD','LOCAL']]
        groups={kind+'_'+s:[encode(r,'task',s) for r in read('data/'+file)] for kind,file in [('old','old_validation.json'),('new','../frozen_data/dev_task.json')] for s in ['STANDARD','LOCAL']}
    else:
        task=read(OUT/'task/training.json');assert task['status']=='complete'
        initial=OUT/'task/selected';assert sha(initial/'adapter_model.safetensors')==task['selected_sha256']
        rows=read('frozen_data/train_rewrite.json');encoded=[encode(r,'rewrite') for r in rows]
        groups={'old_rewrite':[encode(r,'rewrite') for r in read('data/old_rewrite_validation.json')],
                'new_rewrite':[encode(r,'rewrite') for r in read('frozen_data/dev_rewrite.json')]}
    initial_hash=sha(initial/'adapter_model.safetensors');model=load(initial,True);params=[p for p in model.parameters() if p.requires_grad]
    frozen=[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
    opt=torch.optim.AdamW(params,lr=R[stage]['learning_rate'],weight_decay=.01,foreach=False)
    total_updates=math.ceil(len(encoded)/8)*R[stage]['max_epochs'];warmup_updates=max(1,math.ceil(.05*total_updates))
    report=dict(status='training',stage=stage,initial_checkpoint=str(initial),initial_sha256=initial_hash,history=[],optimizer_updates=0,
        examples_per_epoch=len(encoded),trainable_parameters=sum(p.numel() for p in params),backbone_trained=False,default_changed=False)
    report.update(arm=R['arm'],gpu=torch.cuda.get_device_name(0),scheduler='5% linear warmup then cosine to zero; per optimizer update',total_updates=total_updates,warmup_updates=warmup_updates,early_stopping=False,lr_history=[],selection_note='Provisional reference-CE selection. All epochs retained; generation review required before any promotion.')
    best=float('inf');stale=0;started=time.monotonic()
    def snapshot(epoch):
        nonlocal best,stale
        scores=validation(model,stage,groups);path=folder/f'epoch_{epoch:02d}';model.save_pretrained(path)
        entry=dict(epoch=epoch,validation=scores,adapter_sha256=sha(path/'adapter_model.safetensors'));report['history'].append(entry)
        if scores['selection_score']<best:
            best=scores['selection_score'];report['selected_epoch']=epoch;stale=0
        else:stale+=1
        report['stale_epochs']=stale;save(folder/'training.json',report)
        # Only validation outputs at intermediate checkpoints; evaluation is reserved until selection.
        validation_rows=read('frozen_data/dev_task.json')[:24]
        if stage=='rewrite':outputs=fixed_rewrites(model,validation_rows)
        else:outputs=[dict(item_id=r['item_id'],english=decode(model,r,'STANDARD'),singlish=decode(model,r,'LOCAL')) for r in validation_rows]
        save(folder/f'validation_epoch_{epoch:02d}.json',outputs)
        print('VALIDATION',stage,epoch,scores,'selected',report['selected_epoch'],flush=True)
    try:
        snapshot(0)
        for epoch in range(1,R[stage]['max_epochs']+1):
            seed(425+epoch);model.train();opt.zero_grad(set_to_none=True)
            indices=torch.randperm(len(encoded),generator=torch.Generator().manual_seed(425+epoch)).tolist();total=0.
            for start in range(0,len(indices),8):
                part=indices[start:start+8];model.train()
                for i in part:
                    v=loss(model,encoded[i]);total+=float(v.detach());(v/len(part)).backward()
                assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
                norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
                upcoming=report['optimizer_updates']+1
                lr=R[stage]['learning_rate']*lr_factor(upcoming,total_updates,warmup_updates)
                for group in opt.param_groups:group['lr']=lr
                opt.step();opt.zero_grad(set_to_none=True);report['optimizer_updates']=upcoming
                report['lr_history'].append(dict(update=upcoming,learning_rate=lr))
                if report['optimizer_updates']%50==0:
                    save(folder/'training.json',report);print('TRAIN',stage,'epoch',epoch,'updates',report['optimizer_updates'],flush=True)
            report['last_epoch_train_ce']=total/len(encoded);snapshot(epoch)
            # Fixed full budget: CE worsening does not stop either arm.
            torch.save(dict(optimizer=opt.state_dict(),epoch=epoch,updates=report['optimizer_updates'],total_updates=total_updates,warmup_updates=warmup_updates,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),folder/'last_optimizer.pt')
        assert report['optimizer_updates']==total_updates and report['lr_history'][-1]['learning_rate']==0.
        assert frozen==[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
        assert sha(initial/'adapter_model.safetensors')==initial_hash
        (folder/'selected').symlink_to(f"epoch_{report['selected_epoch']:02d}",target_is_directory=True)
        report.update(training_status='complete',source_adapter_unchanged=True,frozen_base_parameters_unchanged=True,
            selected_sha256=sha(folder/'selected/adapter_model.safetensors'),elapsed_seconds=time.monotonic()-started)
        del model,params,opt,v;clear();verify()
        if stage=='rewrite':evaluate_pipeline()
        report['status']='complete';save(folder/'training.json',report);print('TRAINING COMPLETE',stage,flush=True)
    except Exception:
        report.update(status='failed',error=traceback.format_exc());save(folder/'training.json',report);raise

@torch.no_grad()
def evaluate_pipeline():
    rows=read('frozen_data/evaluation_all.json');english_path=OUT/'task/selected';english_hash=sha(english_path/'adapter_model.safetensors')
    model=load(english_path);model.load_adapter(OUT/'rewrite/selected',adapter_name='singlish',is_trainable=False)
    pairs=[]
    for r in rows:
        model.set_adapter('default');model.requires_grad_(False);en=decode(model,r,'STANDARD');anchor=en['reply']
        model.set_adapter('singlish');model.requires_grad_(False)
        sg=decode(model,r,'LOCAL_REWRITE',anchor) if anchor.strip() else dict(reply='',status='blocked_empty_anchor',prompt_sha256=None)
        pairs.append(dict(item_id=r['item_id'],scope=r['scope'],context=r['context'],speakers=r['speakers'],message=r['message'],
            english=anchor,singlish=sg['reply'],standard_anchor=anchor,english_status=en['status'],singlish_status=sg['status'],english_prompt_sha256=en['prompt_sha256'],rewrite_prompt_sha256=sg['prompt_sha256']))
        save(OUT/'rewrite/selected_pipeline.json',dict(status='running',outputs=pairs))
    model.set_adapter('singlish');model.requires_grad_(False)
    diagnostics=fixed_rewrites(model,[r for r in rows if r['scope']=='new_model_authored_evaluation'])
    flags=[dict(item_id=r['item_id'],english_status=r['english_status'],singlish_status=r['singlish_status']) for r in pairs if r['english_status']!='complete' or r['singlish_status']!='complete']
    assert english_hash==sha(english_path/'adapter_model.safetensors')
    save(OUT/'rewrite/fixed_reference_rewrites.json',diagnostics)
    save(OUT/'rewrite/selected_pipeline.json',dict(status='complete' if not flags else 'complete_with_flags',outputs=pairs,flags=flags,
        actual_english_anchors_verified=True,english_adapter_unchanged_during_rewrite=True,reference_replies_used_in_normal_inference=False,
        note='Raw new pipeline outputs. No wording repair, reranking or default promotion. Fixed reference rewrites are a separate diagnostic.'))
    del model;clear()

def main():
    global R,TOK,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['hardware','preflight','pipeline'],required=True);parser.add_argument('--arm',choices=['B','C']);a=parser.parse_args()
    R=read('recipe.json');torch.set_num_threads(4);verify();TOK=checkpoint_tokenizer(Path(R['base_path']))
    if a.stage=='hardware':hardware()
    elif a.stage=='preflight':preflight()
    else:
        assert a.arm;arm=a.arm;R.update(arm=arm,**R['arms'][arm]);OUT=ROOT/'outputs'/arm
        assert read('ready.json')['data_manifest_sha256']==sha(ROOT/'frozen_data/manifest.json')
        save(OUT/'run_recipe.json',dict(arm=arm,task=R['task'],rewrite=R['rewrite'],data_manifest_sha256=sha(ROOT/'frozen_data/manifest.json')))
        train('task');train('rewrite');verify()

if __name__=='__main__':main()
