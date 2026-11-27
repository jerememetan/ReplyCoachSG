"""Controlled Singlish-only target refresh; English anchors remain fixed."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_style_v1 as core
import sealion_target_overlay_v1 as helper


def read(p):return json.loads((ROOT/p).read_text())
def save(p,value):core.save(ROOT/p,value)
def digest(model):
    import hashlib
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if 'lora_' in n:h.update(n.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@core.torch.no_grad()
def validation(model,tok,rows):
    model.eval();values=[];weighted=0.;tokens=0
    for row in rows:
        e=core.encode_rewrite(row,tok,384);loss=float(core.encoded_loss(model,e));n=sum(x!=-100 for x in e['labels'][1:])
        values.append(loss);weighted+=loss*n;tokens+=n
    return dict(examples=len(rows),example_mean_ce=sum(values)/len(values),token_ce=weighted/tokens,
        note='Reference loss is a selection diagnostic, not Singlish naturalness or semantic accuracy.')


@core.torch.no_grad()
def evaluate(model,tok,rows,path):
    model.eval();outputs=[]
    for row in rows:
        out=core.decode(model,tok,row)
        outputs.append(dict(item_id=row['item_id'],scope=row.get('scope'),context=row['context'],speakers=row['speakers'],message=row['message'],
            english=row['standard_anchor'],singlish=out['reply'],status=out['status'],reference=row.get('response'),
            anchor_provenance=row.get('anchor_provenance','Fixed existing English target'),
            generated_english=False,reference_used_for_generation=False))
    report=dict(status='complete',outputs=outputs,
        flags=[r['item_id'] for r in outputs if r['status']!='complete'],
        exact_copies=sum(r['english'].strip().casefold()==r['singlish'].strip().casefold() for r in outputs),
        note='Raw fixed-anchor rewrites, no reranking, copy penalty, lexicon or factual repair. Copy counts do not establish quality.')
    save(path,report);return report


def preflight(tok,recipe,groups):
    for name,h in recipe['payload_manifest'].items():assert core.sha(ROOT/name)==h,name
    assert core.sha(ROOT/'protected_team.jsonl')==recipe['protected_team_sha256']
    spec=importlib.util.spec_from_file_location('export',core.BUNDLE/'infer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.verify_files(core.BUNDLE)
    assert core.sha(core.BUNDLE/'adapters/english/adapter_model.safetensors')==recipe['original_english_sha256']
    assert core.sha(core.BUNDLE/'adapters/singlish/adapter_model.safetensors')==recipe['initial_singlish_sha256']
    sys.modules['replycoach.generate.sealion_target_overlay_v1']=helper
    tests=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_sealion_target_overlay_v1.py'))
    assert tests.wasSuccessful() and tests.testsRun==9
    count=0;max_tokens=0
    for name,rows in groups.items():
        for row in rows:
            assert len(row['context'])==len(row['speakers'])
            prompt=tok.encode(core.rewrite_source(row,row['standard_anchor']),add_special_tokens=False)
            assert len(prompt)+64<=384
            without_target={k:v for k,v in row.items() if k!='response'}
            assert core.rewrite_source(without_target,row['standard_anchor'])==core.rewrite_source(row,row['standard_anchor'])
            if 'response' not in row:continue
            e=core.encode_rewrite(row,tok,384);n=e['prompt_tokens']
            assert e['input_ids'][:n]==prompt and all(x==-100 for x in e['labels'][:n])
            assert e['labels'][n:]==e['input_ids'][n:] and e['labels'][-1]==tok.eos_token_id
            count+=1;max_tokens=max(max_tokens,len(e['input_ids']))
    return dict(status='passed',tests_passed=9,encoded_targets=count,max_tokens=max_tokens,
        no_truncation=True,prompt_mask_eos_verified=True,gold_local_excluded_from_inference_prompt=True,
        team_data_unchanged=True,source_bundle_manifest_verified=True,gpu=core.torch.cuda.get_device_name(0))


def test_cases(groups):
    return groups['test']+groups['controls']+groups['demo']+[dict(r,scope='reused_old_validation') for r in groups['legacy']]


def train(arm,tok,recipe,groups):
    folder=ROOT/'outputs'/arm;assert not folder.exists();folder.mkdir(parents=True)
    core.torch.manual_seed(recipe['seed']);core.torch.cuda.manual_seed_all(recipe['seed'])
    model=core.load(trainable=True)
    model.gradient_checkpointing_enable();model.enable_input_require_grads();model.config.use_cache=False
    params=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==10485760 and all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    frozen=[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
    overlay=groups['control_overlay'] if arm=='old_targets_control' else groups['revised_overlay']
    encoded={name:[core.encode_rewrite(r,tok,384) for r in rows] for name,rows in [('replay',groups['replay']),('common',groups['common']),('overlay',overlay)]}
    optimizer=core.torch.optim.AdamW(params,lr=1e-5,weight_decay=.01,foreach=False)
    report=dict(status='training',arm=arm,optimizer_steps=0,initial_digest=digest(model),history=[],
        schedule_sha256=core.sha(ROOT/'schedule.json'),english_trained=False,base_trained=False,
        reference_policy='Only overlay target wording differs;20shared new casual pairs in both arms.')
    best=float('inf')
    def snapshot(step):
        nonlocal best
        new=validation(model,tok,groups['validation']);old=validation(model,tok,groups['legacy'])
        path=folder/f'step_{step:04d}';model.save_pretrained(path)
        report['history'].append(dict(step=step,new_validation=new,old_validation=old,sha256=core.sha(path/'adapter_model.safetensors')))
        if new['example_mean_ce']<best:best=new['example_mean_ce'];report['selected_step']=step
        evaluate(model,tok,groups['validation'],folder/f'validation_{step:04d}.json')
        save(folder/'training.json',report);print('Validation',arm,step,new,'oldCE',old['token_ce'],flush=True)
    try:
        snapshot(0)
        for step,entry in enumerate(read('schedule.json'),1):
            core.torch.manual_seed(recipe['seed']*1000+step);core.torch.cuda.manual_seed_all(recipe['seed']*1000+step)
            model.train();optimizer.zero_grad(set_to_none=True)
            batch=[encoded['replay'][i] for i in entry['replay']]+[encoded['common'][entry['common']]]+[encoded['overlay'][i] for i in entry['overlay']]
            assert len(batch)==8
            for item in batch:
                v=core.encoded_loss(model,item);assert core.torch.isfinite(v);(v/8).backward()
            assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
            norm=core.torch.nn.utils.clip_grad_norm_(params,1.);assert core.torch.isfinite(norm)
            optimizer.step();optimizer.zero_grad(set_to_none=True);report['optimizer_steps']=step
            if step%20==0:save(folder/'training.json',report);print('Training',arm,step,'/240',flush=True)
            if step in recipe['snapshots'][1:]:snapshot(step)
        assert frozen==[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
        report.update(training_status='complete',frozen_base_unchanged=True,example_exposures=dict(replay=1200,common=240,overlay=480))
        (folder/'selected').symlink_to(f"step_{report['selected_step']:04d}",target_is_directory=True)
        save(folder/'training.json',report)
        del model,params,optimizer,v;gc.collect();core.torch.cuda.empty_cache()
        for label,step in [('selected',report['selected_step']),('final',240)]:
            if label=='final' and report['selected_step']==240:report['final_reuses_selected']=True;continue
            model=core.load(folder/f'step_{step:04d}')
            evaluate(model,tok,test_cases(groups),folder/(label+'_evaluation.json'))
            del model;gc.collect();core.torch.cuda.empty_cache()
        assert core.sha(core.BUNDLE/'adapters/english/adapter_model.safetensors')==recipe['original_english_sha256']
        assert core.sha(core.BUNDLE/'adapters/singlish/adapter_model.safetensors')==recipe['initial_singlish_sha256']
        report.update(status='complete',source_english_and_singlish_unchanged=True,default_changed=False)
        save(folder/'training.json',report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc());save(folder/'training.json',report);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-baseline',action='store_true');parser.add_argument('--arm',choices=['old_targets_control','revised_targets']);args=parser.parse_args()
    assert Path.home().name=='hm.teoh.2024';core.torch.set_num_threads(4)
    recipe=read('recipe.json');groups={name:read('data/'+name+'.json') for name in ['replay','control_overlay','revised_overlay','common','validation','test','controls','legacy','demo']}
    tok=core.checkpoint_tokenizer(core.BUNDLE/'base');check=preflight(tok,recipe,groups)
    assert 'L40S' in check['gpu']
    if args.preflight_baseline:
        assert not (ROOT/'preflight.json').exists();save('preflight.json',check)
        model=core.load();evaluate(model,tok,groups['validation']+test_cases(groups),'outputs/original/evaluation.json')
        print('PREFLIGHT AND ORIGINAL REWRITES COMPLETE',flush=True)
    else:
        assert args.arm and read('outputs/original/evaluation.json')['status']=='complete'
        train(args.arm,tok,recipe,groups)


if __name__=='__main__':main()
