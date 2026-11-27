"""Train useful alternative replies; no abstention gate or oracle conditioning."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import random
import sys
import traceback
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_turn_repair_v1 as core
import sealion_creative_v1 as data
import sealion_creative_sampling_v1 as sampling
from replycoach.generate.sealion_rewrite_v1 import encode_rewrite

ARMS=['single_reference','multiple_references','multiple_plus_contrast']
read,save,sha=core.read,core.save,core.sha


def encode_refs(rows,tok):
    return [[core.encode(dict(r,positive=a['english']),tok) for a in r['alternatives']] for r in rows]


@core.torch.no_grad()
def validation(model,ordinary,references):
    model.eval();total=0.;tokens=0
    for row in ordinary:
        n=sum(x!=-100 for x in row['labels'][1:]);total+=float(core.loss(model,row))*n;tokens+=n
    # Every context has equal weight, irrespective of reference count/length.
    multi=sum(sum(float(core.loss(model,r)) for r in refs)/len(refs) for refs in references)/len(references)
    old=total/tokens
    return dict(ordinary_token_ce=old,new_context_mean_reference_ce=multi,selection_score=.5*(old+multi),
        note='Teacher-forced validation selection, not useful-reply or naturalness accuracy.')


def cases(recipe,groups):
    regressions=random.Random(recipe['seed']).sample(groups['legacy'],20)
    return groups['challenge']+groups['natural']+groups['diagnostics']+[dict(r,scope='reused_old_validation') for r in regressions]


@core.torch.no_grad()
def evaluate(model,tok,rows,path):
    core.activate(model,'default');outputs=[];sampled=[]
    for row in rows:
        decoded=core.decode(model,tok,row)
        outputs.append(dict(item_id=row['item_id'],scope=row.get('scope'),context=row['context'],speakers=row['speakers'],message=row['message'],
            criterion=row.get('criterion'),english=decoded['reply'],english_status=decoded['status'],
            reference_used_for_generation=False,refusal_gate_used=False))
        if row.get('scope')!='reused_old_validation':
            prompt=tok.encode(core.source(row,'STANDARD'),add_special_tokens=False)
            candidates=[sampling.sample_decode(model,tok,prompt,core.BOUNDARY,data.case_seed(row['item_id'])+i) for i in range(3)]
            sampled.append(dict(item_id=row['item_id'],context=row['context'],speakers=row['speakers'],message=row['message'],
                criterion=row.get('criterion'),candidates=candidates,
                unique_english=len({data.normalize_reply(c['reply']) for c in candidates}),
                note='Three unranked stochastic drafts, not three verified answers. No gold target or reference-based selection.'))
    core.activate(model,'singlish')
    for row,out in zip(rows,outputs,strict=True):
        decoded=core.decode(model,tok,row,out['english']) if out['english'].strip() else dict(reply='',status='blocked_empty_anchor')
        out.update(singlish=decoded['reply'],singlish_status=decoded['status'],standard_anchor=out['english'])
    for row in sampled:
        for candidate in row['candidates']:
            decoded=core.decode(model,tok,row,candidate['reply']) if candidate['reply'].strip() else dict(reply='',status='blocked_empty_anchor')
            candidate.update(singlish=decoded['reply'],singlish_status=decoded['status'],standard_anchor=candidate['reply'])
    result=dict(status='complete',outputs=outputs,sampled=sampled,
        flags=[r['item_id'] for r in outputs if r['english_status']!='complete' or r['singlish_status']!='complete'],
        sampled_status_counts={s:sum(c['status']==s for r in sampled for c in r['candidates']) for s in ['complete','length_limit','failed_empty']},
        no_rule_layer=True,no_refusal_gate=True,frozen_original_singlish=True,
        quality_note='Review usefulness, contradictions/roles, creative alternatives and cross-style meaning. No independent human accuracy or automatic reranking.')
    save(path,result);return result


def preflight(tok,recipe,groups):
    for name,h in recipe['payload_manifest'].items():assert sha(ROOT/name)==h,name
    spec=importlib.util.spec_from_file_location('export',core.BUNDLE/'infer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.verify_files(core.BUNDLE)
    assert sha(core.BUNDLE/'adapters/english/adapter_model.safetensors')==recipe['source_english_sha256']
    assert sha(core.BUNDLE/'adapters/singlish/adapter_model.safetensors')==recipe['frozen_singlish_sha256']
    assert sha(ROOT/'protected_team.jsonl')==recipe['team_original_sha256']
    assert (ROOT/'source_records.jsonl').read_bytes().startswith((ROOT/'protected_team.jsonl').read_bytes())
    source=[json.loads(x) for x in (ROOT/'source_records.jsonl').read_text().splitlines() if x.strip()]
    assert len(source)==790
    assert groups['ordinary']==[dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],positive=r['replies']['standard']) for r in source]
    sys.modules['replycoach.generate.sealion_creative_v1']=data
    sys.modules['replycoach.generate.sealion_creative_sampling_v1']=sampling
    tests=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_sealion_creative_v1.py'))
    assert tests.wasSuccessful() and tests.testsRun==9
    count=0;max_length=0
    def check(e):
        nonlocal count,max_length
        n=e['prompt_tokens'];assert all(x==-100 for x in e['labels'][:n])
        assert e['labels'][n:]==e['input_ids'][n:] and e['labels'][-1]==tok.eos_token_id
        assert n+64<=384 and len(e['input_ids'])<=384
        count+=1;max_length=max(max_length,len(e['input_ids']))
    for split in ['train','validation','challenge']:
        for row in groups[split]:
            data.audit(row)
            for alt in row['alternatives']:
                check(core.encode(dict(row,positive=alt['english']),tok))
                check(encode_rewrite(dict(row,standard_anchor=alt['english'],response=alt['singlish']),tok))
            check(core.encode(row,tok,'negative'))
    for row in groups['ordinary']+groups['legacy']:check(core.encode(row,tok))
    for row in cases(recipe,groups):assert len(tok.encode(core.source(row,'STANDARD'),add_special_tokens=False))+64<=384
    return dict(status='passed',tests_passed=9,encoded_targets=count,max_encoded_tokens=max_length,
        prompt_mask_eos_no_truncation=True,original790_unchanged=True,team_data_unchanged=True,
        source_bundle_manifest_verified=True,gpu=core.torch.cuda.get_device_name(0))


def initialize(recipe):
    core.torch.manual_seed(recipe['seed']);core.torch.cuda.manual_seed_all(recipe['seed'])
    model=core.load(trainable=True);model.gradient_checkpointing_enable();model.enable_input_require_grads();model.config.use_cache=False
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==10485760
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    return model


def train(arm,tok,recipe,groups):
    folder=ROOT/'outputs'/arm
    assert not folder.exists();folder.mkdir(parents=True)
    model=initialize(recipe);params=[p for p in model.parameters() if p.requires_grad]
    frozen=[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
    ordinary=[core.encode(r,tok) for r in groups['ordinary']]
    positives=encode_refs(groups['train'],tok);negatives=[core.encode(r,tok,'negative') for r in groups['train']]
    oldvalid=[core.encode(r,tok) for r in groups['legacy']];newvalid=encode_refs(groups['validation'],tok)
    optimizer=core.torch.optim.AdamW(params,lr=1e-5,weight_decay=.01,foreach=False)
    report=dict(status='training',arm=arm,initial_digest=core.digest(model),history=[],optimizer_steps=0,
        schedule_sha256=sha(ROOT/'schedule.json'),contrast_weight=.1 if arm=='multiple_plus_contrast' else 0.,
        reference_visits=[[0,0,0] for _ in groups['train']])
    best=float('inf')
    def snapshot(step):
        nonlocal best
        scores=validation(model,oldvalid,newvalid);checkpoint=folder/f'step_{step:04d}';model.save_pretrained(checkpoint)
        report['history'].append(dict(step=step,metrics=scores,sha256=sha(checkpoint/'adapter_model.safetensors')))
        if scores['selection_score']<best:best=scores['selection_score'];report['selected_step']=step
        save(folder/'training.json',report);print('Validation',arm,step,scores,flush=True)
    try:
        snapshot(0)
        for step,entry in enumerate(read('schedule.json'),1):
            core.torch.manual_seed(recipe['seed']*1000+step);core.torch.cuda.manual_seed_all(recipe['seed']*1000+step)
            model.train();optimizer.zero_grad(set_to_none=True)
            for i in entry['ordinary']:
                value=core.loss(model,ordinary[i]);assert core.torch.isfinite(value);(value/8).backward()
            for i,v in zip(entry['targeted'],entry['variants'],strict=True):
                v=0 if arm=='single_reference' else v
                positive=core.loss(model,positives[i][v]);negative=core.loss(model,negatives[i])
                assert core.torch.isfinite(positive) and core.torch.isfinite(negative)
                contrast=core.torch.relu(.2+positive-negative)
                (positive/8+report['contrast_weight']*contrast/4).backward()
                report['reference_visits'][i][v]+=1
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            norm=core.torch.nn.utils.clip_grad_norm_(params,1.);assert core.torch.isfinite(norm)
            optimizer.step();optimizer.zero_grad(set_to_none=True);report['optimizer_steps']=step
            if step%10==0:save(folder/'training.json',report);print('Training',arm,step,'/300',flush=True)
            if step in recipe['snapshots'][1:]:snapshot(step)
        assert frozen==[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
        assert all(v==([12,0,0] if arm=='single_reference' else [4,4,4]) for v in report['reference_visits'])
        report.update(training_status='complete',frozen_base_unchanged=True,final_digest=core.digest(model))
        (folder/'selected').symlink_to(f"step_{report['selected_step']:04d}",target_is_directory=True)
        save(folder/'training.json',report)
        del model,params,optimizer,value,positive,negative,contrast;gc.collect();core.torch.cuda.empty_cache()
        for label,step in [('selected',report['selected_step']),('final',300)]:
            if label=='final' and step==report['selected_step']:report['final_reuses_selected']=True;continue
            model=core.load(folder/f'step_{step:04d}');model.load_adapter(core.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
            evaluate(model,tok,cases(recipe,groups),folder/(label+'_evaluation.json'))
            del model;gc.collect();core.torch.cuda.empty_cache()
        assert sha(core.BUNDLE/'adapters/english/adapter_model.safetensors')==recipe['source_english_sha256']
        assert sha(core.BUNDLE/'adapters/singlish/adapter_model.safetensors')==recipe['frozen_singlish_sha256']
        report.update(status='complete',source_adapters_unchanged=True,default_changed=False)
        save(folder/'training.json',report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc());save(folder/'training.json',report);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS);parser.add_argument('--preflight-baseline',action='store_true');args=parser.parse_args()
    assert Path.home().name=='hm.teoh.2024'
    core.torch.set_num_threads(4);recipe=read('recipe.json')
    groups={name:read('data/'+name+'.json') for name in ['train','validation','challenge','ordinary','legacy','natural','diagnostics']}
    tok=core.checkpoint_tokenizer(core.BUNDLE/'base');check=preflight(tok,recipe,groups)
    assert 'L40S' in check['gpu']
    if args.preflight_baseline:
        assert not (ROOT/'preflight.json').exists()
        save('preflight.json',check)
        model=core.load();model.load_adapter(core.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
        evaluate(model,tok,cases(recipe,groups),'outputs/original/evaluation.json')
        print('PREFLIGHT AND ORIGINAL BASELINE COMPLETE',flush=True)
    else:
        assert args.arm and read('preflight.json')['status']=='passed'
        assert read('outputs/original/evaluation.json')['status']=='complete'
        train(args.arm,tok,recipe,groups)


if __name__=='__main__':main()
