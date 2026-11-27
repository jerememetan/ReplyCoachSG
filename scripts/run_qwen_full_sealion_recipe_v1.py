"""Replay SEA-LION's public-data pass on Qwen, then its eight-epoch C recipe."""
import argparse
import collections
import json
import math
import time
import traceback

import torch

import run_decoder_compare_v1 as shared
from replycoach.generate.decoder_compare_v1 import write_json
from replycoach.generate.qwen_public_v1 import restore_state, resume_state
from replycoach.generate.sealion_data import encode_reply

ROOT = shared.ROOT


def public_data(recipe):
    values = {}
    for split in ('train', 'validation'):
        entry = recipe['public']['sources'][split]
        path = shared.Path(entry['path'])
        if shared.sha(path) != entry['sha256']:
            raise ValueError('Public source changed: ' + split)
        values[split] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(values[split]) != entry['rows']:
            raise ValueError('Wrong public row count')
    train_groups = {(r['corpus'], r['group_id']) for r in values['train']}
    val_groups = {(r['corpus'], r['group_id']) for r in values['validation']}
    if train_groups & val_groups:
        raise ValueError('Public train/validation group overlap')
    if any(r['style'] != 'DIALOGUE' for rows in values.values() for r in rows):
        raise ValueError('Wrong public style tag')
    return values


def encoded_public(recipe, tok):
    data = public_data(recipe)
    limit = recipe['public']['max_total_tokens']
    train = [encode_reply(r, tok, limit) for r in data['train']]
    validation = {corpus: [encode_reply(r, tok, limit) for r in data['validation'] if r['corpus'] == corpus]
                  for corpus in ('cosem', 'dailydialog')}
    return data, train, validation


@torch.no_grad()
def public_ce(model, groups):
    model.eval()
    scores = {}
    for name, examples in groups.items():
        total = tokens = 0
        for e in examples:
            n = sum(x != -100 for x in e['labels'][1:])
            total += float(shared.loss(model, e)) * n
            tokens += n
        scores[name] = dict(token_ce=total / tokens, examples=len(examples), tokens=tokens)
    return scores


def preflight(recipe):
    # Do the full token-budget check before allocating time to the public pass.
    tok = shared.tokenizer(recipe, 'qwen')
    _, train, validation = encoded_public(recipe, tok)
    checks = dict(train_rows=len(train), validation_rows=sum(map(len, validation.values())),
                  max_tokens=max(len(e['input_ids']) for e in train), no_truncation=True,
                  source_hashes_verified=True, group_disjoint=True)
    longest = max(train, key=lambda e: len(e['input_ids']))
    del train, validation
    shared.prepare(recipe)
    model = shared.model_for_training(recipe, 'qwen')
    model.train()
    torch.cuda.reset_peak_memory_stats()
    value = shared.loss(model, longest)
    value.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    if not torch.isfinite(value) or not any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in params):
        raise ValueError('Longest public example did not backpropagate')
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
        raise ValueError('Longest public example has nonfinite gradients')
    checks.update(longest_forward_backward_finite=True, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    del model, params, value
    shared.clear()
    ready = shared.read('ready.json')
    ready['public_data'] = checks
    write_json(ROOT / 'ready.json', ready)


def public_train(recipe):
    shared.verify(recipe)
    ready = shared.read('ready.json')
    if ready['status'] != 'ready' or not ready.get('public_data', {}).get('longest_forward_backward_finite'):
        raise ValueError('Hardware/data preflight required')
    folder = ROOT / 'outputs/public'
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / 'training.json'
    if report_path.exists() and shared.read('outputs/public/training.json').get('status') == 'complete':
        raise FileExistsError('Completed public adapter is preserved')
    tok = shared.tokenizer(recipe, 'qwen')
    data, train, validation = encoded_public(recipe, tok)
    shared.seed(recipe['seed'])
    model = shared.model_for_training(recipe, 'qwen')
    frozen = [(name, id(p), p._version) for name, p in model.named_parameters() if 'lora_' not in name]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=.01, foreach=False)
    recipe_sha = shared.sha(ROOT / 'recipe.json')
    report = dict(status='running', recipe_sha256=recipe_sha, examples_seen=0, optimizer_updates=0,
                  total_examples=len(train), total_updates=math.ceil(len(train) / 16),
                  trainable_parameters=sum(p.numel() for p in params), source_exposures={}, loss_sum=0.,
                  history=[], selection='Final full public pass; validation monitors only',
                  schedule='Constant5e-5, matching historical SEA-LION public adaptation; warmup/cosine applies to later task/rewrite stages only')
    resume_path = folder / 'public_resume.pt'
    def save_resume():
        temporary = folder / 'public_resume.tmp'
        torch.save(resume_state(model, optimizer, report), temporary)
        temporary.replace(resume_path)
        write_json(report_path, report)
    started = time.monotonic()
    try:
        if resume_path.exists():
            state = torch.load(resume_path, map_location='cpu', weights_only=False)
            if state['report']['recipe_sha256'] != recipe_sha:
                raise ValueError('Resume recipe changed')
            report = restore_state(model, optimizer, state)
            if not 0 <= report['examples_seen'] <= len(train) or (report['examples_seen'] % 16 and report['examples_seen'] != len(train)):
                raise ValueError('Resume is not at an optimizer boundary')
            del state
            print('RESUME', report['examples_seen'], report['optimizer_updates'], flush=True)
        else:
            report['history'].append(dict(examples_seen=0, validation=public_ce(model, validation)))
            save_resume()
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(1425)).tolist()
        seen = collections.Counter(report['source_exposures'])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index in range(report['examples_seen'], len(order)):
            which = order[index]
            group_size = min(16, len(order) - (index // 16) * 16)
            value = shared.loss(model, train[which])
            report['loss_sum'] += float(value.detach())
            (value / group_size).backward()
            seen[data['train'][which]['corpus']] += 1
            count = index + 1
            if count % 16 == 0 or count == len(order):
                norm = torch.nn.utils.clip_grad_norm_(params, 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Public gradient norm is nonfinite')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                report['optimizer_updates'] += 1
            if count % 10000 == 0 or count == len(order):
                report['history'].append(dict(examples_seen=count, validation=public_ce(model, validation)))
                model.train()
            if count % 4000 == 0 or count == len(order):
                report.update(examples_seen=count, source_exposures=dict(seen), elapsed_seconds=time.monotonic() - started)
                save_resume()
                print('PUBLIC', count, '/', len(order), 'updates', report['optimizer_updates'], flush=True)
        if report['optimizer_updates'] != report['total_updates']:
            raise ValueError('Public pass incomplete')
        if frozen != [(name, id(p), p._version) for name, p in model.named_parameters() if 'lora_' not in name]:
            raise ValueError('Frozen Qwen base changed')
        destination = folder / 'final_adapter'
        model.save_pretrained(destination)
        report.update(status='complete', final_adapter_sha256=shared.sha(destination / 'adapter_model.safetensors'),
                      frozen_base_parameters_unchanged=True, completed_full_pass=True)
        write_json(report_path, report)
        print('PUBLIC COMPLETE', report['optimizer_updates'], flush=True)
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        write_json(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preflight', 'public', 'pipeline'], required=True)
    args = parser.parse_args()
    recipe = shared.read('recipe.json')
    torch.set_num_threads(4)
    shared.seed(recipe['seed'])
    if args.stage == 'preflight':
        preflight(recipe)
    elif args.stage == 'public':
        public_train(recipe)
    else:
        shared.train(recipe, 'qwen')


if __name__ == '__main__':
    main()
