"""Matched English-adapter repairs, with frozen challenge and Singlish adapter."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
BUNDLE = Path.home()/'ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1'
sys.path[:0] = [str(ROOT), str(BUNDLE/'src'), str(BUNDLE/'deps')]
import torch
from peft import PeftModel
from replycoach.generate.sealion_v1 import SEALionCausalLM, config_for_checkpoint, checkpoint_tokenizer
from replycoach.generate.sealion_data import encode_reply, source
from replycoach.generate.sealion_dual_v1 import reply_loss, BOUNDARY
from replycoach.generate.sealion_rewrite_v1 import rewrite_source, BOUNDARY as LOCAL_BOUNDARY
from replycoach.generate.sealion_beam_v1 import beam_decode
import sealion_style_v1 as nbest


def read(name):
    return json.loads((ROOT/name).read_text())


def save(name, value):
    path = ROOT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path=None, trainable=False):
    model, info = SEALionCausalLM.from_pretrained(BUNDLE/'base',
        config=config_for_checkpoint(BUNDLE/'base'), dtype=torch.bfloat16,
        device_map={'':'cuda'}, local_files_only=True, output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    assert sum(p.numel() for p in model.parameters()) == 3178255360
    model = PeftModel.from_pretrained(model, path or BUNDLE/'adapters/english', is_trainable=trainable)
    if not trainable:
        model.requires_grad_(False)
    return model


def encode(row, tok, key='positive'):
    return encode_reply(dict(row, response=row[key], style='STANDARD'), tok, 384)


def loss(model, encoded):
    values = {key:torch.tensor([encoded[key]], device='cuda') for key in ['input_ids','attention_mask','labels']}
    labels = values.pop('labels')
    return reply_loss(model(**values, use_cache=False).logits, labels)


@torch.no_grad()
def validation(model, ordinary, pairs):
    model.eval()
    total = 0.; count = 0
    for item in ordinary:
        n = sum(x != -100 for x in item['labels'][1:])
        total += float(loss(model, item))*n; count += n
    positive = [float(loss(model, a)) for a,b in pairs]
    negative = [float(loss(model, b)) for a,b in pairs]
    return dict(ordinary_token_ce=total/count, targeted_positive_mean_ce=sum(positive)/len(positive),
        targeted_negative_mean_ce=sum(negative)/len(negative),
        contrast_wins=sum(a < b for a,b in zip(positive, negative, strict=True)), contrasts=len(pairs),
        note='Teacher-forced diagnostics, not generated-reply accuracy.')


def activate(model, name):
    model.set_adapter(name)
    model.requires_grad_(False)
    model.eval()


@torch.no_grad()
def decode(model, tok, row, anchor=None):
    prompt = source(row, 'STANDARD') if anchor is None else rewrite_source(row, anchor)
    return beam_decode(model, tok, tok.encode(prompt, add_special_tokens=False),
        BOUNDARY if anchor is None else LOCAL_BOUNDARY, num_beams=4,
        no_repeat_ngram_size=3, max_new_tokens=64, length_penalty=1.)


@torch.no_grad()
def evaluate(model, tok, cases, name, adapter='default', expanded=False):
    activate(model, adapter)
    report = dict(status='running', checkpoint=name, outputs=[],
        scope='24 frozen synthetic challenge,1 user diagnostic,20 reused validation regressions; not independent human test.',
        inference=dict(num_beams=4,no_repeat_ngram_size=3,max_new_tokens=64))
    for i,row in enumerate(cases):
        out = decode(model, tok, row)
        result = dict(item_id=row['item_id'],scope=row['scope'],family=row.get('family'),
            context=row['context'],speakers=row['speakers'],message=row['message'],
            english=out['reply'],english_status=out['status'],criterion=row.get('criterion'),
            reference=row.get('positive'),negative=row.get('negative'))
        report['outputs'].append(result)
        if expanded and row['scope']=='user_diagnostic':
            result['english_nbest8'] = nbest.beam_nbest(model,tok,tok.encode(source(row,'STANDARD'),add_special_tokens=False),BOUNDARY,num_beams=8,returns=8)
        if (i+1)%15 == 0:
            save(f'outputs/{name}/evaluation.json', report)
            print('English evaluation',name,i+1,flush=True)
    activate(model, 'singlish')
    for i,(row,result) in enumerate(zip(cases,report['outputs'],strict=True)):
        out = decode(model,tok,row,result['english'])
        result.update(singlish=out['reply'],singlish_status=out['status'],standard_anchor=result['english'])
        if row['scope']=='user_diagnostic':
            anchor="I'll send you a message explaining what I wanted to discuss."
            oracle=decode(model,tok,row,anchor)
            result['assistant_anchor_diagnostic']=dict(anchor=anchor,singlish=oracle['reply'],status=oracle['status'],
                note='Manually supplied appropriate English draft; not English model output.')
        if (i+1)%15 == 0:
            save(f'outputs/{name}/evaluation.json', report)
    report.update(status='complete', cases=len(cases),
        flags=[r['item_id'] for r in report['outputs'] if r['english_status']!='complete' or r['singlish_status']!='complete'])
    save(f'outputs/{name}/evaluation.json',report)
    return report


def digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if 'lora_' in name:
            h.update(name.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def preflight(tok, recipe, groups):
    for name,expected in recipe['data_sha256'].items():
        assert sha(ROOT/'data'/name)==expected
    spec=importlib.util.spec_from_file_location('bundle_infer',BUNDLE/'infer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_files(BUNDLE)
    assert sha(BUNDLE/'adapters/english/adapter_model.safetensors')==recipe['initial_english_sha256']
    assert sha(BUNDLE/'adapters/singlish/adapter_model.safetensors')==recipe['frozen_singlish_sha256']
    sizes={};checked=0
    for name,rows in groups.items():
        lengths=[]
        for row in rows:
            assert len(row['context'])==len(row['speakers'])
            p=tok.encode(source(row,'STANDARD'),add_special_tokens=False)
            assert len(p)+64<=384
            for key in ['positive','negative']:
                if key not in row:continue
                e=encode(row,tok,key);n=e['prompt_tokens']
                assert e['input_ids'][:n]==p
                assert all(x==-100 for x in e['labels'][:n])
                assert e['labels'][n:]==e['input_ids'][n:]
                assert e['labels'][-1]==tok.eos_token_id
                assert all(x!=-100 for x in e['labels'][n:])
                lengths.append(len(e['input_ids']));checked+=1
        sizes[name]=max(lengths,default=0)
    grouped={}
    for row in groups['train']:
        grouped.setdefault(json.dumps([row['context'],row['speakers']]),[]).append(row)
    assert len(grouped)==96
    assert all(len(rows)==2 and rows[0]['message']!=rows[1]['message'] and rows[0]['positive']!=rows[1]['positive'] for rows in grouped.values())
    a=torch.tensor(2.,requires_grad=True);b=torch.tensor(1.,requires_grad=True)
    torch.relu(.2+a-b).backward()
    assert a.grad==1 and b.grad==-1
    checks=dict(status='passed',encoded_targets_checked=checked,max_tokens=sizes,
        paired_latest_question_groups=96,prompt_mask_eos_checks=True,margin_gradient_direction=True,
        source_bundle_manifest_verified=True,source_adapter_checksums_verified=True,
        device=torch.cuda.get_device_name(0),bf16_supported=torch.cuda.is_bf16_supported())
    assert checks['bf16_supported']
    save('preflight.json',checks)
    print('Preflight',json.dumps(checks),flush=True)


def train_arm(arm,tok,groups,recipe,cases):
    torch.manual_seed(recipe['seed']);torch.cuda.manual_seed_all(recipe['seed'])
    model=load(trainable=True)
    model.gradient_checkpointing_enable();model.enable_input_require_grads();model.config.use_cache=False
    params=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==10485760
    assert all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)
    frozen=[(name,id(p),p._version) for name,p in model.named_parameters() if 'lora_' not in name]
    ordinary=[encode(r,tok) for r in groups['ordinary']]
    targeted=[(encode(r,tok),encode(r,tok,'negative')) for r in groups['train']]
    valid=[encode(r,tok) for r in groups['legacy']]
    pairs=[(encode(r,tok),encode(r,tok,'negative')) for r in groups['validation']]
    initial=validation(model,valid,pairs)
    rng=random.Random(recipe['seed'])
    def schedule(size,needed):
        result=[]
        while len(result)<needed:
            order=list(range(size));rng.shuffle(order);result.extend(order)
        return result[:needed]
    count=recipe['optimizer_steps']
    order=dict(ordinary=schedule(len(ordinary),count),targeted=schedule(len(targeted),count*3))
    report=dict(status='training',arm=arm,optimizer_steps=0,selected_step=0,
        initial_adapter_digest=digest(model),initial_validation=initial,
        schedule_sha256=hashlib.sha256(json.dumps(order,sort_keys=True).encode()).hexdigest(),
        margin_weight=0. if arm=='supervised' else .1,history=[dict(step=0,scores=initial)],
        trainable_parameters=10485760,source_english_adapter=recipe['initial_english_sha256'])
    folder=ROOT/'outputs'/arm;folder.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(folder/'selected')
    best=initial['targeted_positive_mean_ce']
    optimizer=torch.optim.AdamW(params,lr=recipe['learning_rate'],weight_decay=.01,foreach=False)
    optimizer.zero_grad(set_to_none=True)
    for step in range(count):
        torch.manual_seed(recipe['seed']*1000+step);torch.cuda.manual_seed_all(recipe['seed']*1000+step)
        model.train()
        old=loss(model,ordinary[order['ordinary'][step]])
        assert torch.isfinite(old)
        (.25*old).backward()
        for index in order['targeted'][step*3:(step+1)*3]:
            preferred=loss(model,targeted[index][0]);assert torch.isfinite(preferred)
            (.25*preferred).backward()
        index=order['targeted'][step*3]
        a=loss(model,targeted[index][0]);b=loss(model,targeted[index][1])
        margin=report['margin_weight']*torch.relu(.2+a-b)
        assert torch.isfinite(margin)
        margin.backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        report['optimizer_steps']=step+1
        if (step+1)%10==0:
            report['latest_losses']=dict(ordinary=float(old.detach()),targeted=float(preferred.detach()),
                margin=float(margin.detach()),grad_norm=float(norm))
            save(f'outputs/{arm}/training.json',report)
            print('Training',arm,step+1,report['latest_losses'],flush=True)
        if step+1 in [40,80,120]:
            scores=validation(model,valid,pairs)
            eligible=scores['ordinary_token_ce']<=initial['ordinary_token_ce']*1.05
            selected=eligible and scores['targeted_positive_mean_ce']<best
            model.save_pretrained(folder/f'step_{step+1:04d}')
            if selected:
                best=scores['targeted_positive_mean_ce'];report['selected_step']=step+1
                model.save_pretrained(folder/'selected')
            report['history'].append(dict(step=step+1,scores=scores,retention_guard_passed=eligible,selected=selected))
            save(f'outputs/{arm}/training.json',report)
        del old,preferred,a,b,margin
    assert frozen==[(name,id(p),p._version) for name,p in model.named_parameters() if 'lora_' not in name]
    report.update(status='trained',final_adapter_digest=digest(model),frozen_base_unchanged=True,
        ordinary_exposures=count,targeted_exposures=count*3,negative_exposures=count,
        selected_sha256=sha(folder/'selected/adapter_model.safetensors'))
    assert report['final_adapter_digest']!=report['initial_adapter_digest']
    save(f'outputs/{arm}/training.json',report)
    del optimizer,params
    # Freeze before adding inference adapters; they never enter the optimizer.
    model.requires_grad_(False)
    model.load_adapter(BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
    model.load_adapter(folder/'selected',adapter_name='selected',is_trainable=False)
    selected=evaluate(model,tok,cases,arm+'/selected',adapter='selected',expanded=True)
    if report['selected_step']==120:
        final=selected
    else:
        final=evaluate(model,tok,cases,arm+'/final120',adapter='default')
    report.update(status='complete',selected_evaluation_flags=selected['flags'],final_evaluation_flags=final['flags'])
    save(f'outputs/{arm}/training.json',report)
    del model;gc.collect();torch.cuda.empty_cache()
    return report


def main():
    assert Path.home().name=='hm.teoh.2024'
    torch.set_num_threads(4)
    recipe=read('recipe.json')
    groups={name:read('data/'+name+'.json') for name in ['train','validation','challenge','ordinary','legacy','diagnostic']}
    tok=checkpoint_tokenizer(BUNDLE/'base')
    preflight(tok,recipe,groups)
    regression=random.Random(recipe['seed']).sample(groups['legacy'],20)
    cases=[dict(r,scope='fresh_synthetic_challenge') for r in groups['challenge']]
    cases += [dict(r,scope='user_diagnostic') for r in groups['diagnostic']]
    cases += [dict(r,scope='reused_development_regression') for r in regression]
    save('evaluation_cases.json',cases)
    started=time.monotonic()
    model=load();model.load_adapter(BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
    baseline=evaluate(model,tok,cases,'baseline',expanded=True)
    assert not baseline['flags']
    del model;gc.collect();torch.cuda.empty_cache()
    report=dict(status='training',recipe=recipe,baseline_complete=True,arms={},gpu=torch.cuda.get_device_name(0))
    save('results.json',report)
    for arm in recipe['arms']:
        report['arms'][arm]=train_arm(arm,tok,groups,recipe,cases)
        save('results.json',report)
    assert report['arms']['supervised']['initial_adapter_digest']==report['arms']['contrastive']['initial_adapter_digest']
    assert report['arms']['supervised']['schedule_sha256']==report['arms']['contrastive']['schedule_sha256']
    assert sha(BUNDLE/'adapters/english/adapter_model.safetensors')==recipe['initial_english_sha256']
    assert sha(BUNDLE/'adapters/singlish/adapter_model.safetensors')==recipe['frozen_singlish_sha256']
    report.update(status='complete',seconds=time.monotonic()-started,source_adapters_unchanged=True,
        matched_initial_weights_and_schedules=True,default_changed=False,human_review_pending=True)
    save('results.json',report)
    print('COMPLETE',json.dumps({k:v['selected_step'] for k,v in report['arms'].items()}),flush=True)


if __name__=='__main__':
    main()
