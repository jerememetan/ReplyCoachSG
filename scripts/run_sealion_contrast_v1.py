"""Matched replay/control versus low-weight contrast repair; original checkpoints read only."""
import argparse
import gc
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import torch
from peft import PeftModel
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
from replycoach.generate.sealion_dual_v1 import item_rows, generate_reply, BOUNDARY
from replycoach.generate.sealion_data import encode_reply, source
from replycoach.generate.sealion_rewrite_v1 import encode_rewrite, rewrite_source, BOUNDARY as REWRITE_BOUNDARY
from replycoach.generate.sealion_beam_v1 import beam_decode
from replycoach.generate.sealion_contrast_v1 import encoded_pair, example_loss, contrast_loss, eligible, selection_score
from run_sealion_dual_v1 import ROOT, OUT, load_base
from run_sealion_dialogue_v1 import enable_training, parameters, validation, update, step

R = {}


def read(path):
    return json.loads((ROOT/path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save():
    OUT.mkdir(exist_ok=True)
    p = OUT/'results.tmp'
    p.write_text(json.dumps(R, indent=2)+'\n')
    p.replace(OUT/'results.json')


def records(path):
    return [json.loads(line) for line in (ROOT/path).read_text().splitlines() if line.strip()]


def digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if 'lora_' in name:
            h.update(name.encode()); h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_sources():
    for p, h in R['recipe']['files_sha256'].items():
        if sha(ROOT/p) != h:
            raise ValueError('Frozen source changed: '+p)
    for stage, info in R['recipe']['initial_adapters'].items():
        for name, h in info['files_sha256'].items():
            if sha(Path(info['checkpoint'])/name) != h:
                raise ValueError('Original '+stage+' adapter changed')


@torch.no_grad()
def scores(model, replay, auxiliary):
    ordinary = validation(model, {'replay': replay})['groups']['replay']
    model.eval(); pos, neg = [], []
    for a, b in auxiliary:
        pos.append(float(example_loss(model, a)))
        neg.append(float(example_loss(model, b)))
    p, n = sum(pos)/len(pos), sum(neg)/len(neg)
    return dict(replay=ordinary, positive_example_mean_ce=p, negative_example_mean_ce=n,
                ranking_win_fraction=sum(a < b for a, b in zip(pos, neg))/len(pos),
                selection_score=selection_score(ordinary['token_ce'], p))


def clear():
    gc.collect(); torch.cuda.empty_cache()


def load(stage, checkpoint=None, trainable=False):
    base, problems = load_base()
    if any(problems.values()):
        raise ValueError(problems)
    return PeftModel.from_pretrained(base, checkpoint or R['recipe']['initial_adapters'][stage]['checkpoint'], is_trainable=trainable)


@torch.no_grad()
def generate(model, row, stage, method, anchor=None):
    if method == 'greedy':
        return generate_reply(model, TOK, row, 'STANDARD') if stage == 'english' else generate_rewrite(model, TOK, row, anchor)
    ids = TOK.encode(source(row, 'STANDARD') if stage == 'english' else rewrite_source(row, anchor), add_special_tokens=False)
    output = beam_decode(model, TOK, ids, BOUNDARY if stage == 'english' else REWRITE_BOUNDARY,
                         num_beams=4, no_repeat_ngram_size=3, length_penalty=1.)
    if stage == 'rewrite':
        output['standard_anchor'] = anchor
    return output


def generate_rewrite(model, tok, row, anchor):
    from replycoach.generate.sealion_rewrite_v1 import generate_rewrite as f
    return f(model, tok, row, anchor)


def baseline_english(model):
    prior = read('baseline_beam_results.json')['methods']
    result = {}
    for method in ['greedy', 'beam4_ng3']:
        outputs = {r['item_id']: r['styles']['STANDARD'] for r in prior[method]['direct']}
        # Verify the inherited model/decoder on two saved ordinary conversations.
        for row in CASES[:2]:
            out = generate(model, row, 'english', method)
            if out['reply'] != outputs[row['item_id']]['reply']:
                raise ValueError('Initial English exact replay failed')
        for row in CASES:
            if row['item_id'] not in outputs:
                outputs[row['item_id']] = generate(model, row, 'english', method)
        result[method] = [dict(item_id=row['item_id'], output=outputs[row['item_id']]) for row in CASES]
    R['baseline_english'] = result
    save()


def evaluate_english(model):
    R['outputs']['english'] = {}
    for method in ['greedy', 'beam4_ng3']:
        rows = R['outputs']['english'][method] = []
        for row in CASES:
            rows.append(dict(item_id=row['item_id'], output=generate(model, row, 'english', method)))
            save()


def baseline_rewrite(model):
    prior = read('baseline_beam_results.json')['methods']
    R['baseline_rewrite'] = {}
    for method in ['greedy', 'beam4_ng3']:
        outputs = {r['item_id']: r['rewrite'] for r in prior[method]['rewrite']['own_english']}
        anchors = {r['item_id']: r['output']['reply'] for r in R['baseline_english'][method]}
        for row in CASES[:2]:
            out = generate(model, row, 'rewrite', method, anchors[row['item_id']])
            if out['reply'] != outputs[row['item_id']]['reply']:
                raise ValueError('Initial renderer exact replay failed')
        for row in CASES:
            if row['item_id'] not in outputs:
                outputs[row['item_id']] = generate(model, row, 'rewrite', method, anchors[row['item_id']])
            if outputs[row['item_id']]['standard_anchor'] != anchors[row['item_id']]:
                raise ValueError('Baseline anchor mismatch')
        R['baseline_rewrite'][method] = [dict(item_id=row['item_id'], output=outputs[row['item_id']]) for row in CASES]
        save()


def evaluate_rewrite(model):
    R['outputs']['rewrite'] = {}
    for method in ['greedy', 'beam4_ng3']:
        routes = R['outputs']['rewrite'][method] = {}
        own = [r['output']['reply'] for r in R['outputs']['english'][method]]
        fixed = [r['output']['reply'] for r in R['baseline_english'][method]]
        diagnostic = [r for r in CASES if 'diagnostic_anchor' in r]
        for route, cases, anchors in [('own_english', CASES, own), ('fixed_baseline_english', CASES, fixed),
                                      ('diagnostic', diagnostic, [r['diagnostic_anchor'] for r in diagnostic])]:
            results = routes[route] = []
            for row, anchor in zip(cases, anchors, strict=True):
                out = generate(model, row, 'rewrite', method, anchor) if anchor.strip() else dict(reply='', status='failed_empty_anchor')
                results.append(dict(item_id=row['item_id'], standard_anchor=anchor, output=out))
                save()


def train(stage, arm):
    torch.manual_seed(425); torch.cuda.manual_seed_all(425); random.seed(425)
    if stage == 'english':
        from run_sealion_expanded_v1 import expanded_rows
        expanded = records('data/training_expansion/generated600_repair_v1/combined_train.jsonl')
        checked = expanded_rows(expanded, records('data/dev/train.jsonl'), records('data/dev/validation.jsonl'), 'preserved')
        train = [r for r in checked if r['style']=='STANDARD']
        valid = [r for r in item_rows(records('data/dev/validation.jsonl')) if r['style']=='STANDARD']
        encoder = encode_reply
    else:
        from run_sealion_full_rewrite_v1 import pair_rows
        audit = read('data/training_expansion/sealion_rewrite_v1/pair_audit.json')
        def pairs(split, path):
            allowed = {r['id'] for r in audit['items'] if r['split']==split and r['eligible']}
            return pair_rows(records(path), allowed)
        train = pairs('train', 'data/training_expansion/generated600_repair_v1/combined_train.jsonl')
        valid = pairs('validation', 'data/dev/validation.jsonl')
        encoder = encode_rewrite
    expected = (790, 50) if stage=='english' else (745, 42)
    if (len(train), len(valid)) != expected:
        raise ValueError('Replay counts changed')
    train = [encoder(r, TOK, 384) for r in train]
    valid = [encoder(r, TOK, 384) for r in valid]
    aux_train = [encoded_pair(r, TOK, stage) for r in read('data/training_expansion/sealion_contrast_v1/train.json')]
    aux_val = [encoded_pair(r, TOK, stage) for r in read('data/training_expansion/sealion_contrast_v1/validation.json')]
    rng = random.Random(425)
    order = []
    while len(order) < 1600:
        part = list(range(len(train))); rng.shuffle(part); order.extend(part)
    aux_order = []
    while len(aux_order) < 200:
        part = list(range(len(aux_train))); rng.shuffle(part); aux_order.extend(part)
    plan = dict(main=order[:1600], auxiliary=aux_order[:200])
    model = load(stage, trainable=True)
    enable_training(model); params = parameters(model)
    if sum(p.numel() for p in params) != 10485760:
        raise ValueError('Unexpected trainable parameter count')
    phase = R['stages'][stage] = dict(status='running', replay_train=len(train), replay_validation=len(valid),
        auxiliary_train=128, auxiliary_validation=32, trainable_parameters=sum(p.numel() for p in params),
        initial_adapter_sha256=digest(model), schedule_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(), history=[])
    save()
    if stage == 'english': baseline_english(model)
    else: baseline_rewrite(model)
    initial = scores(model, valid, aux_val)
    initial_ce = initial['replay']['token_ce']; best_score = initial['selection_score']; selected_step = 0
    selected = OUT/stage/'selected'
    selected.mkdir(parents=True)
    model.save_pretrained(selected); TOK.save_pretrained(selected)
    phase['history'].append(dict(step=0, scores=initial, replay_guard_passed=True, selected=True))
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.01, foreach=False)
    weight = 0. if arm=='control' else .1
    optimizer.zero_grad(set_to_none=True)
    for index in range(200):
        model.train()
        # Reset per step: matched input order and dropout streams between arms.
        torch.manual_seed(425000+index); torch.cuda.manual_seed_all(425000+index)
        for which in plan['main'][index*8:(index+1)*8]:
            update(model, optimizer, params, train[which], 8)
        before = [p.grad.detach().clone() for p in params] if index==0 else None
        a, b = aux_train[plan['auxiliary'][index]]
        positive = example_loss(model, a); negative = example_loss(model, b)
        auxiliary = weight*contrast_loss(positive, negative)
        if not torch.isfinite(auxiliary): raise FloatingPointError('Nonfinite auxiliary objective')
        auxiliary.backward()
        if index==0:
            changed = any(not torch.equal(p.grad, old) for p, old in zip(params, before, strict=True))
            if changed != (arm=='repair'): raise ValueError('Auxiliary gradient audit failed')
            phase['auxiliary_changed_primary_gradient'] = changed
            del before
            frozen_ok = all(p.grad is None for p in model.parameters() if not p.requires_grad)
            if not frozen_ok: raise ValueError('Frozen base received gradients')
            phase['frozen_base_received_no_gradients'] = True
        step(optimizer, params)
        count = index+1
        if count in [25, 50, 100, 150, 200]:
            score = scores(model, valid, aux_val)
            passed = eligible(score['replay']['token_ce'], initial_ce)
            chosen = passed and score['selection_score'] < best_score
            if chosen:
                best_score = score['selection_score']; selected_step = count
                model.save_pretrained(selected); TOK.save_pretrained(selected)
            checkpoint = OUT/stage/f'step_{count:04d}'
            model.save_pretrained(checkpoint); TOK.save_pretrained(checkpoint)
            phase['history'].append(dict(step=count, scores=score, replay_guard_passed=passed, selected=chosen))
            phase.update(optimizer_steps=count, selected_step=selected_step, best_score=best_score)
            save(); print(stage, arm, count, score, 'selected', selected_step, flush=True)
    phase.update(status='trained', ordinary_exposures=1600, auxiliary_pair_exposures=200,
                 auxiliary_loss_weight=weight, selected_step=selected_step)
    del optimizer, params, model; clear()
    model = load(stage, selected).eval()
    phase['selected_validation'] = scores(model, valid, aux_val)
    save()
    if stage=='english': evaluate_english(model)
    else: evaluate_rewrite(model)
    phase['status'] = 'complete'; save()
    del model; clear()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--arm', choices=['control','repair'], required=True)
    args = parser.parse_args()
    if (OUT/'results.json').exists(): raise FileExistsError('Fresh snapshot required')
    R.update(status='running', arm=args.arm, recipe=read('recipe.json'), stages={}, outputs={}, default_changed=False)
    verify_sources()
    download = read('download_manifest.json')
    if any(download[k] != R['recipe'][k] for k in ['revision','weights_sha256']):
        raise ValueError('Pinned SEA-LION base identity mismatch')
    if 'L40S' not in torch.cuda.get_device_name(0): raise ValueError('L40S required')
    torch.set_num_threads(4)
    global TOK, CASES
    TOK = checkpoint_tokenizer(ROOT/'model')
    CASES = sum((read(p) for p in ['data/evaluation/sealion_full_rewrite_v1/inputs.json', 'data/evaluation/restored_pipeline_v1/inputs.json', 'data/evaluation/context_pairs_v1/inputs.json']), [])
    if len(CASES)!=126 or len({r['item_id'] for r in CASES})!=126: raise ValueError('Evaluation coverage')
    started = time.monotonic(); save()
    try:
        train('english', args.arm); train('rewrite', args.arm)
        verify_sources()
        flags = []
        for method, rows in R['outputs']['english'].items():
            flags += [dict(stage='english', method=method, item_id=r['item_id'], status=r['output']['status']) for r in rows if r['output']['status']!='complete']
        for method, routes in R['outputs']['rewrite'].items():
            for route, rows in routes.items():
                flags += [dict(stage='rewrite', route=route, method=method, item_id=r['item_id'], status=r['output']['status']) for r in rows if r['output']['status']!='complete']
        R.update(status='complete_with_flags' if flags else 'complete', flags=flags, original_adapters_unchanged=True,
                 seconds=time.monotonic()-started, quality_review='Pending contextual/meaning/naturalness review; loss and synthetic ranking are not quality certification.')
        save()
    except Exception as exc:
        R.update(status='failed', error=str(exc), traceback=traceback.format_exc(), seconds=time.monotonic()-started)
        save(); raise


if __name__=='__main__': main()
