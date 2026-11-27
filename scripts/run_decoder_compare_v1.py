"""Shared decoder task/rewrite pipeline with optional public-adapted initialization."""
import argparse
import gc
import hashlib
import json
import math
import random
import time
import traceback
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from replycoach.generate.decoder_compare_v1 import (
    encode, lr_multiplier, reorder_cache, validate_groups, write_json,
)
from replycoach.generate.sealion_audited_restart_v1 import prompt
from replycoach.generate.sealion_beam_v1 import banned_next, rendered
from replycoach.generate.sealion_dual_v1 import reply_loss
from replycoach.generate.sealion_rewrite_v1 import BOUNDARY
from replycoach.generate.sealion_v1 import (
    SEALionCausalLM, checkpoint_loading_problems, checkpoint_tokenizer,
    config_for_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed(value):
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def tokenizer(recipe, arm):
    path = recipe['models'][arm]['path']
    tok = checkpoint_tokenizer(path) if arm == 'sealion' else AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    return tok


def base_model(recipe, arm):
    path = recipe['models'][arm]['path']
    kwargs = dict(dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True,
                  output_loading_info=True)
    if arm == 'sealion':
        model, info = SEALionCausalLM.from_pretrained(path, config=config_for_checkpoint(path), **kwargs)
    else:
        model, info = AutoModelForCausalLM.from_pretrained(path, attn_implementation='eager', **kwargs)
    problems = checkpoint_loading_problems(info)
    if any(problems.values()) or any(p.is_meta for p in model.parameters()):
        raise ValueError('Strict model loading failed: ' + str(problems))
    return model


def model_for_training(recipe, arm, initial=None):
    seed(recipe['seed'])
    base = base_model(recipe, arm)
    if initial is None:
        model = get_peft_model(base, LoraConfig(
            task_type='CAUSAL_LM', r=8, lora_alpha=16, lora_dropout=.05,
            target_modules=recipe['models'][arm]['lora_modules']))
    else:
        model = PeftModel.from_pretrained(base, initial, is_trainable=True)
        expected = load_file(str(Path(initial) / 'adapter_model.safetensors'))
        actual = get_peft_model_state_dict(model)
        if set(expected) != set(actual) or any(not torch.equal(actual[k].detach().cpu(), v) for k, v in expected.items()):
            raise ValueError('Loaded adapter differs from its saved source')
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name for name, _ in trainable):
        raise ValueError('Only adapter parameters may train')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def loss(model, e):
    values = {k: torch.tensor([e[k]], device='cuda') for k in ('input_ids', 'attention_mask')}
    labels = torch.tensor([e['labels']], device='cuda')
    result = reply_loss(model(**values, use_cache=False).logits, labels)
    if not torch.isfinite(result):
        raise FloatingPointError('Nonfinite reply loss')
    return result


@torch.no_grad()
def validation(model, groups):
    model.eval()
    metrics = {}
    for name, examples in groups.items():
        total = tokens = 0
        for e in examples:
            n = sum(t != -100 for t in e['labels'][1:])
            total += float(loss(model, e)) * n
            tokens += n
        metrics[name] = dict(token_ce=total / tokens, examples=len(examples), tokens=tokens)
    old = sum(v['token_ce'] for k, v in metrics.items() if k.startswith('old_')) / sum(k.startswith('old_') for k in metrics)
    development = sum(v['token_ce'] for k, v in metrics.items() if k.startswith('development_')) / sum(k.startswith('development_') for k in metrics)
    return dict(groups=metrics, selection_score=.5 * (old + development))


@torch.no_grad()
def decode(model, tok, row, style, anchor=None):
    """Same beam search for both legacy SEA-LION and Qwen's DynamicCache."""
    text = prompt(row, style, anchor)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) + 64 > 384:
        return dict(reply='', status='blocked_source_budget', prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
    model.eval()
    current = torch.tensor([ids], device='cuda')
    past = None
    active, finished = [([], 0.)], []
    def score(c):
        return c[1] / max(1, len(c[0]))
    for position in range(64):
        mask = torch.ones((len(active), len(ids) + position), device='cuda', dtype=torch.long)
        output = model(input_ids=current, attention_mask=mask, past_key_values=past, use_cache=True)
        logits = output.logits[:, -1, :].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite decoding logits')
        probabilities = torch.log_softmax(logits, dim=-1)
        for i, (tokens, _) in enumerate(active):
            banned = banned_next(tokens, 3)
            if banned:
                probabilities[i, list(banned)] = -torch.inf
        cumulative = probabilities + torch.tensor([v for _, v in active], device='cuda')[:, None]
        width = cumulative.shape[-1]
        values, indices = torch.topk(cumulative.flatten(), min(8, cumulative.numel()))
        next_active, parents = [], []
        for value, index in zip(values.tolist(), indices.tolist()):
            if not math.isfinite(value):
                continue
            parent, token = divmod(index, width)
            tokens = active[parent][0] + [token]
            _, _, reason = rendered(tok, tokens, BOUNDARY)
            if reason or position == 63:
                finished.append((tokens, value, reason or 'length_limit'))
            elif len(next_active) < 4:
                next_active.append((tokens, value))
                parents.append(parent)
        finished = sorted(finished, key=score, reverse=True)[:4]
        if not next_active or (finished and score(finished[0]) >= max(v / 64 for _, v in next_active)):
            break
        past = reorder_cache(output.past_key_values, torch.tensor(parents, device='cuda'))
        active = next_active
        current = torch.tensor([[tokens[-1]] for tokens, _ in active], device='cuda')
    if not finished:
        raise RuntimeError('No decoded candidates')
    tokens, value, termination = max(finished, key=score)
    raw, reply, _ = rendered(tok, tokens, BOUNDARY)
    reply = reply.strip()
    return dict(reply=reply, raw_text=raw, termination=termination, generated_tokens=len(tokens),
                status='failed_empty' if not reply else 'length_limit' if termination == 'length_limit' else 'complete',
                prompt_sha256=hashlib.sha256(text.encode()).hexdigest())


def verify(recipe):
    for path, digest in recipe['payload_manifest'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Payload changed: ' + path)
    models = read('models/download_manifest.json')
    for path, digest in models['files'].items():
        if sha(path) != digest:
            raise ValueError('Base file changed: ' + path)
    data = {key: read('data/' + key + '.json') for key in
            ('train_task', 'train_rewrite', 'old_validation', 'development', 'evaluation', 'reference_evaluation')}
    validate_groups(data['train_task'], data['train_rewrite'], data['old_validation'],
                    data['development'], data['evaluation'], data['reference_evaluation'])
    if len(data['train_task']) != 768 or len(data['train_rewrite']) != 766:
        raise ValueError('Unexpected training count')
    return data


def prepare(recipe):
    """On-device strict load, finite backward and cache equivalence before student jobs."""
    data = verify(recipe)
    results = {}
    for arm in recipe['models']:
        tok = tokenizer(recipe, arm)
        all_encoded = [encode(r, tok, 'task', s) for key in ('train_task', 'old_validation', 'development') for r in data[key] for s in ('STANDARD', 'LOCAL')]
        all_encoded += [encode(r, tok, 'rewrite') for key in ('train_rewrite', 'development', 'reference_evaluation') for r in data[key]]
        model = model_for_training(recipe, arm)
        e = encode(data['train_task'][0], tok, 'task', 'STANDARD')
        v = loss(model, e)
        v.backward()
        params = [p for p in model.parameters() if p.requires_grad]
        if not any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in params):
            raise ValueError('No adapter gradients')
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
            raise ValueError('Nonfinite gradient')
        if any(p.grad is not None for name, p in model.named_parameters() if 'lora_' not in name):
            raise ValueError('Backbone gradient')
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            ids = torch.tensor([e['input_ids'][:min(12, len(e['input_ids']))]], device='cuda')
            full = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, -1, :].float()
            first = model(input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]), use_cache=True)
            cached = model(input_ids=ids[:, -1:], attention_mask=torch.ones_like(ids), past_key_values=first.past_key_values, use_cache=True).logits[:, -1, :].float()
            if not torch.allclose(full, cached, atol=.25, rtol=.03):
                raise ValueError('Cached/full logits differ')
        base_count = sum(p.numel() for p in model.get_base_model().parameters()) - sum(p.numel() for p in params)
        if base_count != recipe['models'][arm]['pretrained_parameters']:
            raise ValueError('Unexpected base parameter count')
        with model.disable_adapter():
            baseline = [dict(item_id=r['item_id'], english=r['standard'],
                             singlish=decode(model, tok, r, 'LOCAL_REWRITE', r['standard']))
                        for r in data['reference_evaluation']]
        write_json(ROOT / 'baseline' / (arm + '_fixed_reference24.json'),
                   dict(outputs=baseline, scope='Base-only zero-shot prompt diagnostic; no adapters or task training, not an exhaustive language-capability test'))
        results[arm] = dict(forward_backward_finite=True, strict_load=True, cached_logits_verified=True,
                            trainable_parameters=sum(p.numel() for p in params), base_parameters=base_count,
                            max_encoded_tokens=max(len(x['input_ids']) for x in all_encoded),
                            gpu=torch.cuda.get_device_name(0))
        del model, params, v, full, first, cached
        clear()
    write_json(ROOT / 'ready.json', dict(status='ready', checks=results, torch=torch.__version__))


def train(recipe, arm):
    data = verify(recipe)
    ready = read('ready.json')
    if ready['status'] != 'ready':
        raise ValueError('Preflight not passed')
    tok = tokenizer(recipe, arm)
    out = ROOT / 'outputs' / arm
    if out.exists():
        raise FileExistsError('Existing arm outputs; preserve rather than overwrite')
    out.mkdir(parents=True)
    for stage in ('task', 'rewrite'):
        folder = out / stage
        folder.mkdir()
        rows = data['train_task'] if stage == 'task' else data['train_rewrite']
        examples = [encode(r, tok, stage, s) for r in rows for s in ('STANDARD', 'LOCAL')] if stage == 'task' else [encode(r, tok, stage) for r in rows]
        old_rows = data['old_validation'] if stage == 'task' else read('data/old_rewrite_validation.json')
        groups = {}
        for name, group in [('old', old_rows), ('development', data['development'])]:
            if stage == 'task':
                groups.update({name + '_' + s: [encode(r, tok, stage, s) for r in group] for s in ('STANDARD', 'LOCAL')})
            else:
                groups[name + '_rewrite'] = [encode(r, tok, stage) for r in group]
        initial = None if stage == 'task' else out / 'task' / 'selected'
        if stage == 'task' and recipe['training'].get('initial_task_adapter'):
            public = read('outputs/public/training.json')
            initial = ROOT / recipe['training']['initial_task_adapter']
            if public['status'] != 'complete' or sha(initial / 'adapter_model.safetensors') != public['final_adapter_sha256']:
                raise ValueError('Public stage not complete or handoff changed')
        model = model_for_training(recipe, arm, initial)
        frozen = [(name, id(p), p._version) for name, p in model.named_parameters() if 'lora_' not in name]
        params = [p for p in model.parameters() if p.requires_grad]
        peak_lr = recipe['training'][stage + '_learning_rate']
        optimizer = torch.optim.AdamW(params, lr=peak_lr, weight_decay=.01, foreach=False)
        total_updates = math.ceil(len(examples) / 8) * recipe['training']['epochs']
        warmup = max(1, math.ceil(total_updates * .05))
        report = dict(status='training', arm=arm, stage=stage, history=[], optimizer_updates=0,
                      total_updates=total_updates, warmup_updates=warmup,
                      initial='fresh_base_zero_initialized_LoRA' if initial is None else str(initial),
                      trainable_parameters=sum(p.numel() for p in params),
                      checkpoint_selection='Provisional within-model reference CE; not a cross-model quality score.',
                      fixed_budget_checkpoint=f"epoch_{recipe['training']['epochs']:02d}", training_loss_history=[])
        best = float('inf')
        def snapshot(epoch):
            nonlocal best
            scores = validation(model, groups)
            path = folder / f'epoch_{epoch:02d}'
            model.save_pretrained(path)
            report['history'].append(dict(epoch=epoch, validation=scores, adapter_sha256=sha(path / 'adapter_model.safetensors')))
            if scores['selection_score'] < best:
                best = scores['selection_score']
                report['selected_epoch'] = epoch
            write_json(folder / 'training.json', report)
            outputs = []
            for r in data['development']:
                if stage == 'task':
                    outputs.append(dict(item_id=r['item_id'], english=decode(model, tok, r, 'STANDARD'), local_direct=decode(model, tok, r, 'LOCAL')))
                else:
                    outputs.append(dict(item_id=r['item_id'], english=r['standard'], singlish=decode(model, tok, r, 'LOCAL_REWRITE', r['standard'])))
            write_json(folder / f'validation_epoch_{epoch:02d}.json', outputs)
            print('VALIDATION', arm, stage, epoch, scores['selection_score'], flush=True)
        try:
            snapshot(0)
            for epoch in range(1, recipe['training']['epochs'] + 1):
                seed(recipe['seed'] + epoch)
                order = torch.randperm(len(examples), generator=torch.Generator().manual_seed(recipe['seed'] + epoch)).tolist()
                total_loss = 0.
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, len(order), 8):
                    part = order[start:start + 8]
                    for i in part:
                        v = loss(model, examples[i])
                        total_loss += float(v.detach())
                        (v / len(part)).backward()
                    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite gradient norm')
                    update = report['optimizer_updates'] + 1
                    for group in optimizer.param_groups:
                        group['lr'] = peak_lr * lr_multiplier(update, total_updates, warmup)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    report['optimizer_updates'] = update
                    if update % 50 == 0:
                        write_json(folder / 'training.json', report)
                        print('TRAIN', arm, stage, epoch, update, '/', total_updates, flush=True)
                report['training_loss_history'].append(dict(epoch=epoch, example_mean_ce=total_loss / len(examples)))
                snapshot(epoch)
                model.train()
            if frozen != [(name, id(p), p._version) for name, p in model.named_parameters() if 'lora_' not in name]:
                raise ValueError('Frozen backbone mutated')
            if report['optimizer_updates'] != total_updates:
                raise ValueError('Incomplete budget')
            (folder / 'selected').symlink_to(f"epoch_{report['selected_epoch']:02d}", target_is_directory=True)
            report.update(status='complete', frozen_base_parameters_unchanged=True)
            write_json(folder / 'training.json', report)
        except Exception:
            report.update(status='failed', error=traceback.format_exc())
            write_json(folder / 'training.json', report)
            raise
        del model, params, optimizer, v
        clear()
    evaluate(recipe, arm, 'selected')
    evaluate(recipe, arm, f"epoch_{recipe['training']['epochs']:02d}")


@torch.no_grad()
def evaluate(recipe, arm, checkpoint):
    data = verify(recipe)
    folder = ROOT / 'outputs' / arm
    tok = tokenizer(recipe, arm)
    model = PeftModel.from_pretrained(base_model(recipe, arm), folder / 'task' / checkpoint, is_trainable=False)
    model.load_adapter(folder / 'rewrite' / checkpoint, adapter_name='singlish', is_trainable=False)
    rows = []
    destination = folder / f'pipeline_{checkpoint}.json'
    for row in data['evaluation']:
        model.set_adapter('default')
        english = decode(model, tok, row, 'STANDARD')
        model.set_adapter('singlish')
        local = decode(model, tok, row, 'LOCAL_REWRITE', english['reply']) if english['reply'] else dict(reply='', status='blocked_empty_anchor')
        rows.append(dict(item_id=row['item_id'], context=row['context'], speakers=row['speakers'], message=row['message'], english=english, singlish=local, standard_anchor=english['reply']))
        write_json(destination, dict(status='running', outputs=rows))
    flags = [dict(item_id=r['item_id'], english=r['english']['status'], singlish=r['singlish']['status']) for r in rows if r['english']['status'] != 'complete' or r['singlish']['status'] != 'complete']
    write_json(destination, dict(status='complete_with_flags' if flags else 'complete', outputs=rows, flags=flags,
                               scope='75 reused development cases; not independent heldout evaluation'))
    model.set_adapter('singlish')
    fixed = [dict(item_id=r['item_id'], english=r['standard'], singlish=decode(model, tok, r, 'LOCAL_REWRITE', r['standard'])) for r in data['reference_evaluation']]
    write_json(folder / f'fixed_reference_{checkpoint}.json', fixed)
    del model
    clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preflight', 'train'], required=True)
    parser.add_argument('--arm', choices=['sealion', 'qwen'])
    args = parser.parse_args()
    recipe = read('recipe.json')
    torch.set_num_threads(4)
    seed(recipe['seed'])
    started = time.monotonic()
    if args.stage == 'preflight':
        prepare(recipe)
    elif args.arm:
        train(recipe, args.arm)
    else:
        parser.error('--arm required for training')
    print('COMPLETE', args.stage, args.arm, 'seconds', time.monotonic() - started, flush=True)


if __name__ == '__main__':
    main()
