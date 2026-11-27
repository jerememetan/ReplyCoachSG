"""Matched selective-abstention SFT from original English LoRA, no input rules."""
import gc
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_turn_repair_v1 as core
import sealion_abstention_v1 as abstain


def read(name):return json.loads((ROOT/name).read_text())
def save(name,value):core.save(name,value)


@core.torch.no_grad()
def english_evaluation(model,tok,rows,path,adapter=None):
    if adapter is not None:core.activate(model,adapter)
    model.eval();outputs=[]
    for row in rows:
        decoded=core.decode(model,tok,row)
        outputs.append(dict(row,english=decoded['reply'],english_status=decoded['status'],
            canonical_abstention=abstain.canonical_refusal(decoded['reply']),uncertainty_cue=abstain.uncertainty_cue(decoded['reply'])))
    result=dict(status='complete',outputs=outputs,metrics=abstain.metrics(outputs))
    save(path,result)
    return result


@core.torch.no_grad()
def ordinary_ce(model,encoded):
    model.eval();total=0.;count=0
    for item in encoded:
        n=sum(x!=-100 for x in item['labels'][1:])
        total+=float(core.loss(model,item))*n;count+=n
    return total/count


@core.torch.no_grad()
def paired_evaluation(model,tok,rows,path,adapter='default'):
    result=english_evaluation(model,tok,rows,path,adapter)
    core.activate(model,'singlish')
    for row in result['outputs']:
        if row['canonical_abstention']:
            row.update(singlish=abstain.REFUSAL,singlish_status='abstained',needs_more_context=True,
                singlish_provenance='English-model-emitted canonical abstention; rewriting bypassed')
        else:
            out=core.decode(model,tok,row,row['english'])
            row.update(singlish=out['reply'],singlish_status=out['status'],needs_more_context=False,
                singlish_provenance='Original frozen repair25 Singlish adapter')
        row['standard_anchor']=row['english']
    result['flags']=[r['item_id'] for r in result['outputs'] if r['english_status']!='complete' or r['singlish_status'] not in ['complete','abstained']]
    save(path,result)
    return result


def preflight(tok,recipe,groups):
    for name,expected in read('payload_manifest.json').items():assert core.sha(ROOT/name)==expected,name
    spec=importlib.util.spec_from_file_location('export_infer',core.BUNDLE/'infer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.verify_files(core.BUNDLE)
    assert core.sha(core.BUNDLE/'adapters/english/adapter_model.safetensors')==recipe['source_english_sha256']
    assert core.sha(core.BUNDLE/'adapters/singlish/adapter_model.safetensors')==recipe['frozen_singlish_sha256']
    import replycoach.generate
    sys.modules['replycoach.generate.sealion_abstention_v1']=abstain
    tests=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_sealion_abstention_v1.py'))
    assert tests.wasSuccessful() and tests.testsRun==9
    lengths={};targets=0
    for name,rows in groups.items():
        assert core.sha(ROOT/'data'/f'{name}.json')==recipe['data_sha256'][name+'.json']
        sizes=[]
        for row in rows:
            e=core.encode(row,tok);n=e['prompt_tokens']
            assert all(x==-100 for x in e['labels'][:n])
            assert e['labels'][n:]==e['input_ids'][n:] and e['labels'][-1]==tok.eos_token_id
            assert len(e['input_ids'])<=384 and n+64<=384
            assert (row['positive']==abstain.REFUSAL)==(row['requires_refusal'] is True)
            sizes.append(len(e['input_ids']));targets+=1
        lengths[name]=max(sizes)
    check=dict(status='passed',tests_passed=tests.testsRun,encoded_targets=targets,max_tokens=lengths,
        prompt_mask_eos_no_truncation=True,source_bundle_manifest_verified=True,
        gpu=core.torch.cuda.get_device_name(0),bf16_supported=core.torch.cuda.is_bf16_supported())
    assert check['bf16_supported'];save('preflight.json',check)
    print('Preflight',json.dumps(check),flush=True)


def train_arm(arm,tok,groups,recipe,cases,baseline_validation):
    seed=recipe['seed'];core.torch.manual_seed(seed);core.torch.cuda.manual_seed_all(seed)
    model=core.load(trainable=True)
    model.gradient_checkpointing_enable();model.enable_input_require_grads();model.config.use_cache=False
    params=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==10485760
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    frozen=[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
    source=groups['control'] if arm=='restored_context_control' else groups['train']
    encoded=[core.encode(r,tok) for r in source]
    legacy=[core.encode(r,tok) for r in groups['legacy']]
    rng=random.Random(seed);order=[]
    for epoch in range(2):
        indices=list(range(1120));rng.shuffle(indices);order.extend(indices)
    assert len(order)==2240
    folder=ROOT/'outputs'/arm;folder.mkdir(parents=True)
    model.save_pretrained(folder/'selected')
    initial=english_evaluation(model,tok,groups['validation'],f'outputs/{arm}/validation_0000.json')
    assert [r['english'] for r in initial['outputs']]==[r['english'] for r in baseline_validation['outputs']]
    report=dict(status='training',arm=arm,optimizer_steps=0,selected_step=0,
        initial_adapter_digest=core.digest(model),trainable_parameters=10485760,
        schedule_sha256=__import__('hashlib').sha256(json.dumps(order).encode()).hexdigest(),
        initial_validation=initial['metrics'],history=[dict(step=0,metrics=initial['metrics'],ordinary_ce=ordinary_ce(model,legacy))])
    save(f'outputs/{arm}/training.json',report)
    best=initial['metrics']['answerability_balanced_score']
    optimizer=core.torch.optim.AdamW(params,lr=recipe['learning_rate'],weight_decay=.01,foreach=False)
    optimizer.zero_grad(set_to_none=True)
    refusal_exposures=0
    for step in range(280):
        core.torch.manual_seed(seed*1000+step);core.torch.cuda.manual_seed_all(seed*1000+step)
        model.train();total=0.
        for index in order[step*8:(step+1)*8]:
            value=core.loss(model,encoded[index]);assert core.torch.isfinite(value)
            (value/8).backward();total+=float(value.detach())
            refusal_exposures+=source[index]['requires_refusal'] is True
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        norm=core.torch.nn.utils.clip_grad_norm_(params,1.);assert core.torch.isfinite(norm)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        report.update(optimizer_steps=step+1,example_exposures=(step+1)*8,refusal_exposures=refusal_exposures)
        if (step+1)%10==0:
            report['latest_mean_ce']=total/8;save(f'outputs/{arm}/training.json',report)
            print('Training',arm,step+1,'/280','CE',total/8,flush=True)
        if step+1 in [70,140,280]:
            checkpoint=folder/f'step_{step+1:04d}';model.save_pretrained(checkpoint)
            generated=english_evaluation(model,tok,groups['validation'],f'outputs/{arm}/validation_{step+1:04d}.json')
            scores=generated['metrics'];allowed=abstain.eligible(scores,initial['metrics'])
            selected=allowed and scores['answerability_balanced_score']>best
            if selected:
                best=scores['answerability_balanced_score'];report['selected_step']=step+1
                model.save_pretrained(folder/'selected')
            report['history'].append(dict(step=step+1,metrics=scores,ordinary_ce=ordinary_ce(model,legacy),eligible=allowed,selected=selected))
            save(f'outputs/{arm}/training.json',report)
            print('Validation',arm,step+1,json.dumps(scores),'selected',selected,flush=True)
        del value
    assert frozen==[(n,id(p),p._version) for n,p in model.named_parameters() if 'lora_' not in n]
    assert refusal_exposures==(240 if arm=='selective_abstention' else 0)
    report.update(status='trained',frozen_base_unchanged=True,final_adapter_digest=core.digest(model),
        selected_sha256=core.sha(folder/'selected/adapter_model.safetensors'))
    assert report['final_adapter_digest']!=report['initial_adapter_digest']
    save(f'outputs/{arm}/training.json',report)
    del optimizer,params
    model.requires_grad_(False)
    model.load_adapter(core.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
    model.load_adapter(folder/'selected',adapter_name='chosen',is_trainable=False)
    selected=paired_evaluation(model,tok,cases,f'outputs/{arm}/selected/evaluation.json',adapter='chosen')
    final=selected if report['selected_step']==280 else paired_evaluation(model,tok,cases,f'outputs/{arm}/final280/evaluation.json')
    report.update(status='complete',selected_evaluation_flags=selected['flags'],final_evaluation_flags=final['flags'])
    save(f'outputs/{arm}/training.json',report)
    del model;gc.collect();core.torch.cuda.empty_cache()
    return report


def main():
    assert Path.home().name=='hm.teoh.2024'
    core.torch.set_num_threads(4)
    recipe=read('recipe.json')
    groups={name:read('data/'+name+'.json') for name in ['train','control','validation','challenge','legacy']}
    tok=core.checkpoint_tokenizer(core.BUNDLE/'base')
    preflight(tok,recipe,groups)
    diagnostic=dict(item_id='USER-CALL-001',context=['Can u call me now?','Eh i not so free eh?','Ohh, then when can we call?'],
        speakers=['user','peer','user'],message='I am not sure, what do you want to call me about',
        requires_refusal=None,reference_values=[],scope='reused_user_diagnostic')
    cases=groups['challenge']+random.Random(recipe['seed']).sample(groups['legacy'],20)+[diagnostic]
    save('evaluation_cases.json',cases)
    assert not any(r['context']==diagnostic['context'] and r['message']==diagnostic['message'] for r in groups['train'])
    started=time.monotonic()
    model=core.load();model.load_adapter(core.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
    baseline=paired_evaluation(model,tok,cases,'outputs/baseline/evaluation.json')
    core.activate(model,'default')
    validation=english_evaluation(model,tok,groups['validation'],'outputs/baseline/validation.json')
    del model;gc.collect();core.torch.cuda.empty_cache()
    report=dict(status='training',recipe=recipe,baseline_complete=True,arms={},gpu=core.torch.cuda.get_device_name(0))
    save('results.json',report)
    for arm in recipe['arms']:
        report['arms'][arm]=train_arm(arm,tok,groups,recipe,cases,validation)
        save('results.json',report)
    a,b=[report['arms'][arm] for arm in recipe['arms']]
    assert a['initial_adapter_digest']==b['initial_adapter_digest'] and a['schedule_sha256']==b['schedule_sha256']
    assert core.sha(core.BUNDLE/'adapters/english/adapter_model.safetensors')==recipe['source_english_sha256']
    assert core.sha(core.BUNDLE/'adapters/singlish/adapter_model.safetensors')==recipe['frozen_singlish_sha256']
    report.update(status='complete',seconds=time.monotonic()-started,source_adapters_unchanged=True,
        matched_initial_weights_and_slot_schedules=True,default_changed=False,human_review_pending=True)
    save('results.json',report)
    print('COMPLETE',json.dumps({arm:data['selected_step'] for arm,data in report['arms'].items()}),flush=True)


if __name__=='__main__':main()
