"""Warm-start distillation: continue one trained shared task LoRA, never reset it."""
import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import train_distill_mt5_v2 as ce
import train_mt5_soft_kd_v1 as kd
from run_mt5_adapter_switch_v1 import load_base
from run_decoder_compare_v1 import sha
from replycoach.generate.decoder_compare_v1 import lr_multiplier, write_json
from replycoach.generate.distill_mt5_v2 import digest
from replycoach.generate.mt5_matched_v1 import source
from replycoach.generate.mt5_warm_distill_v1 import load_shared_adapter, paired_update_groups, selection_score

ROOT = Path(__file__).resolve().parents[1]
ce.ROOT = ROOT
STAGES = ('english', 'singlish')


def read(path):
    return json.loads(Path(path).read_text())


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def verify_inputs():
    recipe, warm = read(ROOT / 'recipe.json'), read(ROOT / 'warm_recipe.json')
    for name, expected in read(ROOT / 'warm_payload_manifest.json').items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen warm payload changed: ' + name)
    base = ce.verify_base(recipe)
    adapter = Path(warm['initial_adapter']['path'])
    for name, expected in warm['initial_adapter']['files_sha256'].items():
        if sha(adapter / name) != expected or sha(Path(warm['initial_adapter']['source_path']) / name) != expected:
            raise ValueError('Preserved trained adapter changed: ' + name)
    if sha(ROOT / 'targets/full/manifest.json') != warm['target_manifest_sha256']:
        raise ValueError('Matched target manifest changed')
    for name, expected in read(ROOT / 'targets/full/manifest.json')['files'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Reviewed source changed: ' + name)
    kd.ROOT = Path(warm['teacher_cache_root'])
    if sha(kd.ROOT / 'soft_payload_manifest.json') != warm['teacher_cache_payload_sha256']:
        raise ValueError('Original cache experiment identity changed')
    rows = {arm: read(ROOT / f'targets/full/{arm}.json') for arm in ('original', 'sealion', 'qwen')}
    ids = [[r['item_id'] for r in v] for v in rows.values()]
    if len(ids[0]) != 571 or not ids[0] == ids[1] == ids[2]:
        raise ValueError('Expected identical571 matched conversations')
    for arm in ('sealion', 'qwen'):
        for stage in STAGES:
            spec = recipe['teachers'][arm][stage]
            if sha(Path(spec['path']) / 'adapter_model.safetensors') != spec['sha256']:
                raise ValueError('Preserved teacher changed')
    return recipe, warm, base, rows


def warm_model(base, warm, trainable=True, checkpoint=None):
    ce.seed(warm['seed'])
    model = load_shared_adapter(load_base(base), checkpoint or warm['initial_adapter']['path'], trainable)
    initial = ce.adapter_digest(model)
    if checkpoint is None and initial != warm['initial_adapter']['tensor_sha256']:
        raise ValueError('Warm-start tensor identity differs from trained adapter')
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    else:
        model.eval()
    return model, [p for p in model.parameters() if p.requires_grad]


def teacher_caches(recipe, arm, rows):
    records, reports, heads = {}, {}, {}
    for stage in STAGES:
        entries, head, report = kd.load_cache(recipe, arm, stage, rows)
        records[stage], reports[stage] = entries, report
        if stage == 'singlish':
            if report['head_sha256'] != reports['english']['head_sha256']:
                raise ValueError('Teacher stage output heads differ')
            heads[stage] = heads['english']
            del head
        else:
            heads[stage] = head
    return records, heads, reports


def encoded(rows, tok, warm):
    return {s: ce.encoded(rows, tok, s, warm['student']) for s in STAGES}


def validation_rows(tok, warm):
    return {s: ce.encoded(read(ROOT / ('data/old_validation.json' if s == 'english' else 'data/old_rewrite_validation.json')),
                         tok, s, warm['student']) for s in STAGES}


def validation(model, data, tok):
    scores = {s: ce.validation(model, data[s], tok) for s in STAGES}
    scores['selection_score'] = selection_score(scores)
    return scores


def modules(model, records, warm):
    return torch.nn.ModuleDict({s: kd.alignment(model, records[s][0]['hidden'].shape[-1], warm['kd']) for s in STAGES})


def objective(model, projectors, data, records, heads, sh, tok, stage, index, replay=False):
    if projectors is None:
        loss, _ = ce.objective(model, [data[stage][index]], tok)
        return loss, dict(hard_ce=loss.detach(), total=loss.detach())
    return kd.objective(model, projectors[stage], data[stage][index], records[stage][index],
                        tok, sh, heads[stage], replay=replay)


@torch.no_grad()
def diagnostics(model, projectors, data, records, heads, sh, tok):
    model.eval()
    result = {}
    for stage in STAGES:
        totals = {}
        for index in range(len(data[stage])):
            _, metrics = objective(model, projectors, data, records, heads, sh, tok, stage, index, replay=True)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + float(value) / len(data[stage])
        result[stage] = totals
    result['joint_total'] = .5 * sum(result[s]['total'] for s in STAGES)
    return result


def optimizer(params, projectors, warm):
    groups = [dict(params=params, lr=warm['student']['learning_rate'], initial_lr=warm['student']['learning_rate'])]
    if projectors is not None:
        groups.append(dict(params=list(projectors.parameters()), lr=warm['kd']['projector_learning_rate'],
                           initial_lr=warm['kd']['projector_learning_rate']))
    return torch.optim.AdamW(groups, weight_decay=warm['student']['weight_decay'])


@torch.no_grad()
def generate(model, tok, cases, style, warm, anchors=None):
    """Same192/48 beam4 batched50 serialization/decoding contract as old export."""
    model.eval()
    answers = []
    for start in range(0, len(cases), 50):
        part, prompts, ready, batch_answers = cases[start:start + 50], [], [], {}
        for i, row in enumerate(part):
            anchor = None if anchors is None else anchors[start + i]
            if anchors is not None and not anchor.strip():
                batch_answers[i] = dict(item_id=row['item_id'], reply='', status='failed_empty_anchor', token_ids=[])
                continue
            prompts.append(source(row, style, anchor)); ready.append(i)
        if ready:
            inputs = tok(prompts, return_tensors='pt', padding=True)
            if inputs['input_ids'].shape[-1] > warm['student']['source_limit']:
                raise ValueError('No generation truncation')
            output = model.generate(**{k:v.cuda() for k,v in inputs.items()}, max_new_tokens=warm['student']['generation_new_tokens'],
                                    num_beams=4, no_repeat_ngram_size=3, do_sample=False, use_cache=True)
            for i, seq in zip(ready, output, strict=True):
                text = tok.decode(seq, skip_special_tokens=True)
                status = 'failed_empty' if not text.strip() else 'complete' if tok.eos_token_id in seq.tolist() else 'length_limit'
                batch_answers[i] = dict(item_id=part[i]['item_id'], reply=text, status=status, token_ids=seq.tolist())
        answers.extend(batch_answers[i] for i in range(len(part)))
    return answers


def evaluate(model, tok, warm, output):
    initial = ce.adapter_digest(model)
    cases, fixed = read(ROOT / 'data/evaluation.json'), read(ROOT / 'data/reference_evaluation.json')
    english = generate(model, tok, cases, 'STANDARD', warm)
    anchors = [x['reply'] for x in english]
    singlish = generate(model, tok, cases, 'LOCAL_REWRITE', warm, anchors)
    reference = generate(model, tok, fixed, 'LOCAL_REWRITE', warm, [x['standard'] for x in fixed])
    if ce.adapter_digest(model) != initial:
        raise ValueError('Shared adapter changed during inference')
    write_json(output, dict(pipeline=[dict(item_id=c['item_id'], english=e, singlish=s, standard_anchor=e['reply'])
        for c,e,s in zip(cases,english,singlish,strict=True)],
        fixed_reference=[dict(item_id=c['item_id'],standard_anchor=c['standard'],singlish=s)
                         for c,s in zip(fixed,reference,strict=True)],
        shared_adapter_for_both_stages=True, adapter_unchanged_during_inference=True,
        initial_adapter_tensor_sha256=initial, independent_test=False, quality_scored=False))


def baseline(recipe, warm, base, tok):
    output = ROOT / 'baseline'
    if output.exists():raise FileExistsError('Preserve baseline attempt')
    model, _ = warm_model(base, warm, trainable=False)
    inputs = read(ROOT / 'data/reference_replay_inputs.json')
    expected = read(ROOT / 'data/reference_replay_expected.json')
    english = generate(model, tok, inputs, 'STANDARD', warm)
    singlish = generate(model, tok, inputs, 'LOCAL_REWRITE', warm, [x['reply'] for x in english])
    actual = [dict(item_id=c['item_id'],english=e['reply'],singlish=s['reply']) for c,e,s in zip(inputs,english,singlish,strict=True)]
    output.mkdir()
    write_json(output / 'replay.json',dict(passed=actual==expected,expected=expected,actual=actual))
    if actual != expected:raise ValueError('Unchanged trained model failed portable-example replay')
    scores = validation(model, validation_rows(tok,warm),tok)
    evaluate(model,tok,warm,output/'evaluation.json')
    write_json(output/'complete.json',dict(status='complete',source_adapter=warm['initial_adapter'],
        validation=scores,portable_replay_exact=len(actual),shared_adapter=True,adapter_tensor_sha256=ce.adapter_digest(model)))
    del model
    clear()


def pilot(recipe,warm,base,tok,arm,method,rows):
    path=ROOT/f'pilot/{arm}_{method}.json'
    if path.exists():raise FileExistsError('Preserve warm pilot evidence')
    records=heads=reports=None
    if method=='kd':records,heads,reports=teacher_caches(recipe,arm,rows)
    model,params=warm_model(base,warm)
    initial=ce.adapter_digest(model);sh=kd.student_head(model)
    head_initial=kd.tensor_digest(sh.state_dict())
    projectors=modules(model,records,warm) if method=='kd' else None
    data=encoded(rows[:4],tok,warm)
    before=diagnostics(model,projectors,data,records,heads,sh,tok)
    opt=optimizer(params,projectors,warm)
    all_params=params+([] if projectors is None else list(projectors.parameters()))
    lora_grad=projector_grad=False
    for step in range(warm['pilot_updates']):
        model.train();opt.zero_grad(set_to_none=True)
        for stage in STAGES:
            loss,_=objective(model,projectors,data,records,heads,sh,tok,stage,step%4)
            (loss/2).backward()
        lora_grad|=any(p.grad is not None and p.grad.norm()>0 for p in params)
        if projectors is not None:projector_grad|=all(p.grad is not None and p.grad.norm()>0 for p in projectors.parameters())
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen base acquired gradients')
        norm=torch.nn.utils.clip_grad_norm_(all_params,warm['student']['gradient_clip'])
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite warm-pilot gradient')
        opt.step()
    after=diagnostics(model,projectors,data,records,heads,sh,tok)
    unchanged=kd.tensor_digest(sh.state_dict())==head_initial
    passed=after['joint_total']<before['joint_total'] and bool(lora_grad) and unchanged and ce.adapter_digest(model)!=initial
    if projectors is not None:passed=passed and bool(projector_grad)
    report=dict(status='passed' if passed else 'failed',arm=arm,method=method,before=before,after=after,
        initial_adapter_tensor_sha256=initial,source_adapter_file_sha256=warm['initial_adapter']['files_sha256']['adapter_model.safetensors'],
        optimizer_updates=warm['pilot_updates'],nonzero_student_gradients=bool(lora_grad),
        nonzero_projector_gradients=bool(projector_grad) if method=='kd' else None,
        finite_gradients=True,frozen_base_no_gradients=True,base_head_unchanged=unchanged,
        native_head_replay=True,teacher_cache_verified=method=='kd',shared_adapter=True,checkpoint_created=False)
    write_json(path,report)
    if not passed:raise ValueError('Warm continuation pilot failed')
    del opt,all_params,params,model,sh,projectors,records,heads,reports
    clear()
    return report


def fit(recipe,warm,base,tok,arm,method,rows):
    output=ROOT/f'students/{arm}_{method}'
    if output.exists():raise FileExistsError('Preserve prior warm training attempt')
    records=heads=reports=None
    if method=='kd':records,heads,reports=teacher_caches(recipe,arm,rows)
    model,params=warm_model(base,warm)
    initial=ce.adapter_digest(model);sh=kd.student_head(model)
    projectors=modules(model,records,warm) if method=='kd' else None
    data,valid=encoded(rows,tok,warm),validation_rows(tok,warm)
    scores=validation(model,valid,tok)
    original_scores=read(ROOT/'baseline/complete.json')['validation']
    for stage in STAGES:
        if (not math.isclose(scores[stage]['token_ce'],original_scores[stage]['token_ce'],rel_tol=1e-5,abs_tol=1e-6)
                or scores[stage]['tokens']!=original_scores[stage]['tokens']):
            raise ValueError('Warm start validation differs from unchanged baseline')
    settings=warm['student'];total=math.ceil(len(rows)/settings['pairs_per_update'])*settings['epochs']
    report=dict(status='running',arm=arm,method=method,shared_adapter=True,
        training_conversations=len(rows),training_examples=2*len(rows),training_rows_sha256=digest(rows),
        initial_adapter_tensor_sha256=initial,initial_validation=scores,optimizer_updates=0,
        source_adapter=warm['initial_adapter'],student_parameters=sum(p.numel() for p in params),
        projector_parameters=0 if projectors is None else sum(p.numel() for p in projectors.parameters()),
        selected_epoch=0,selected_score=scores['selection_score'],history=[],protocol_sha256=sha(ROOT/'warm_recipe.json'))
    output.mkdir(parents=True)
    model.save_pretrained(output/'epoch_00')
    report['history'].append(dict(epoch=0,validation=scores,optimizer_updates=0,adapter_sha256=sha(output/'epoch_00/adapter_model.safetensors')))
    write_json(output/'training.json',report)
    opt=optimizer(params,projectors,warm)
    all_params=params+([] if projectors is None else list(projectors.parameters()))
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,settings['epochs']+1):
        model.train();sums={s:{} for s in STAGES}
        for group in paired_update_groups(len(rows),warm['seed'],epoch,settings['pairs_per_update']):
            opt.zero_grad(set_to_none=True)
            warmup=max(1,math.ceil(total*settings['warmup_fraction']))
            multiplier=lr_multiplier(report['optimizer_updates']+1,total,warmup)
            for g in opt.param_groups:g['lr']=g['initial_lr']*multiplier
            for stage,index in group:
                loss,metrics=objective(model,projectors,data,records,heads,sh,tok,stage,index)
                (loss/len(group)).backward()
                for key,value in metrics.items():sums[stage][key]=sums[stage].get(key,0.)+float(value)/len(rows)
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen base acquired gradients')
            norm=torch.nn.utils.clip_grad_norm_(all_params,settings['gradient_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite continued-training gradient')
            opt.step();report['optimizer_updates']+=1
        scores=validation(model,valid,tok)
        checkpoint=output/f'epoch_{epoch:02d}';model.save_pretrained(checkpoint)
        if projectors is not None:torch.save(projectors.state_dict(),checkpoint/'training_only_projectors.pt')
        report['history'].append(dict(epoch=epoch,validation=scores,train_metrics=sums,
            optimizer_updates=report['optimizer_updates'],adapter_sha256=sha(checkpoint/'adapter_model.safetensors')))
        if scores['selection_score']<report['selected_score']:
            report.update(selected_epoch=epoch,selected_score=scores['selection_score'])
        write_json(output/'training.json',report)
        print(arm,method,epoch,scores,sums,flush=True)
    if report['optimizer_updates']!=total:raise ValueError('Paired optimizer budget mismatch')
    report.update(status='complete',seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    (output/'selected').symlink_to(f"epoch_{report['selected_epoch']:02d}",target_is_directory=True)
    (output/'final').symlink_to(f"epoch_{settings['epochs']:02d}",target_is_directory=True)
    write_json(output/'training.json',report)
    del opt,all_params,params,model,sh,projectors,records,heads,reports
    clear()
    for label in ['selected','final']:
        model,_=warm_model(base,warm,trainable=False,checkpoint=output/label)
        evaluate(model,tok,warm,output/f'evaluation_{label}.json')
        if label=='selected' and report['selected_epoch']==0:
            if ce.adapter_digest(model)!=initial:raise ValueError('Selected epoch0 weights differ from source adapter')
            baseline=read(ROOT/'baseline/evaluation.json')
            report['selected_epoch0_exact_generation_replay']=(read(output/f'evaluation_{label}.json')==baseline)
            write_json(output/'training.json',report)
        del model
        clear()
    verify_inputs()
    write_json(output/'complete.json',dict(status='complete',arm=arm,method=method,shared_adapter=True,
        initial_adapter_tensor_sha256=initial,selected_epoch=report['selected_epoch'],original_model_preserved=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['gate','fit'],required=True)
    parser.add_argument('--arm',choices=['sealion','qwen'])
    parser.add_argument('--method',choices=['ce','kd'])
    args=parser.parse_args();torch.set_num_threads(4)
    recipe,warm,base,rows=verify_inputs()
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    if args.mode=='gate':
        gate=ROOT/'pilot/gate.json'
        if gate.exists():raise FileExistsError('Preserve previous warm gate')
        # All571 targets and common reference labels must fit original192/48 budgets.
        budgets={}
        for arm in ['sealion','qwen']:
            data=encoded(rows[arm],tok,warm)
            budgets[arm]={s:dict(source_max=max(len(x) for x,y in data[s]),target_max=max(len(y) for x,y in data[s])) for s in STAGES}
        validation_rows(tok,warm)
        baseline(recipe,warm,base,tok)
        report=dict(status='running',protocol_sha256=sha(ROOT/'warm_recipe.json'),budgets=budgets,arms={})
        for arm in ['sealion','qwen']:
            for method in ['ce','kd']:
                report['arms'][arm+'_'+method]=pilot(recipe,warm,base,tok,arm,method,rows[arm]);write_json(gate,report)
        verify_inputs();report.update(status='passed',trained_adapter_preserved=True)
        write_json(gate,report)
        return
    if not args.arm or not args.method:raise ValueError('Arm/method required')
    gate=read(ROOT/'pilot/gate.json')
    if gate['status']!='passed' or gate['protocol_sha256']!=sha(ROOT/'warm_recipe.json'):
        raise ValueError('All four warm continuation pilots must pass')
    fit(recipe,warm,base,tok,args.arm,args.method,rows[args.arm])


if __name__=='__main__':main()
