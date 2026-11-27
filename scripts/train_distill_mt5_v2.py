"""Matched mT5 students on frozen, reviewed paired distillation targets."""
import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from transformers import AutoTokenizer

from run_mt5_adapter_switch_v1 import load_base
from run_decoder_compare_v1 import sha
from replycoach.generate.decoder_compare_v1 import lr_multiplier, write_json
from replycoach.generate.distill_mt5_v2 import digest
from replycoach.generate.mt5_adapter_switch_v1 import new_task_adapter, load_switching_adapters, activate_inference
from replycoach.generate.mt5_matched_v1 import encode, source, reply_loss
from replycoach.generate.train_v5 import collate

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def seed(value):
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def adapter_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(get_peft_model_state_dict(model).items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def train_rows(rows, stage):
    return [dict(**{k: r[k] for k in ('item_id', 'context', 'speakers', 'message')},
                 style='STANDARD', response=r['standard']) if stage == 'english' else
            dict(**{k: r[k] for k in ('item_id', 'context', 'speakers', 'message')},
                 standard_anchor=r['standard'], response=r['local']) for r in rows]


def tensor(rows, tok):
    x, a, y = collate(rows, tok.pad_token_id)
    return x.cuda(), a.cuda(), y.cuda()


def objective(model, batch, tok):
    x, a, y = tensor(batch, tok)
    loss = reply_loss(model(input_ids=x, attention_mask=a, labels=y, use_cache=False).logits, y)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite mT5 loss')
    return loss, int(y.ne(-100).sum())


@torch.no_grad()
def validation(model, rows, tok):
    model.eval()
    total = count = 0
    for row in rows:
        loss, n = objective(model, [row], tok)
        total += float(loss) * n
        count += n
    return dict(token_ce=total / count, tokens=count, examples=len(rows))


def verify_base(recipe):
    manifest = read(ROOT / 'student_base_manifest.json')
    path = Path(recipe['student']['base'])
    for name, entry in manifest['files'].items():
        if sha(path / name) != entry['sha256']:
            raise ValueError('Preserved student base changed: ' + name)
    return path


def new_model(path, recipe):
    seed(recipe['seed'])
    model = new_task_adapter(load_base(path))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != recipe['student']['lora']['trainable_parameters']:
        raise ValueError('Unexpected student adapter size')
    return model, params


def encoded(rows, tok, stage, settings):
    return [encode(r, tok, 'direct' if stage == 'english' else 'rewrite',
                   settings['source_limit'], settings['target_limit']) for r in train_rows(rows, stage)]


def preflight(recipe, base, tok, rows, output):
    """Check real paired-target optimization without saving a usable student."""
    if output.exists():
        raise FileExistsError('Preflight evidence already exists')
    report = dict(status='running', rows=[r['item_id'] for r in rows[:4]], stages={})
    for stage in ('english', 'singlish'):
        model, params = new_model(base, recipe)
        initial = adapter_digest(model)
        data = encoded(rows[:4], tok, stage, recipe['student'])
        before = validation(model, data, tok)
        opt = torch.optim.AdamW(params, lr=recipe['student']['learning_rate'], weight_decay=.01)
        for step in range(24):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss, _ = objective(model, [data[step % len(data)]], tok)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite preflight gradient')
            opt.step()
        after = validation(model, data, tok)
        passed = after['token_ce'] < before['token_ce'] and adapter_digest(model) != initial
        report['stages'][stage] = dict(before=before, after=after, initial_adapter_sha256=initial,
                                       steps=24, finite_gradients=True, passed=passed)
        write_json(output, report)
        if not passed:
            raise ValueError('Student failed paired-target fitting check')
        del opt, params, model
        gc.collect()
        torch.cuda.empty_cache()
    if len({s['initial_adapter_sha256'] for s in report['stages'].values()}) != 1:
        raise ValueError('Independent stage adapters did not initialize identically')
    verify_base(recipe)
    report.update(status='passed', shared_base_unchanged=True, student_checkpoint_created=False)
    write_json(output, report)


def fit(recipe, base, tok, rows, valid, stage, output):
    if output.exists():
        raise FileExistsError('Preserve previous fitting attempt: ' + str(output))
    settings = recipe['student']
    model, params = new_model(base, recipe)
    train = encoded(rows, tok, stage, settings)
    val = encoded(valid, tok, stage, settings)
    report = dict(status='running', stage=stage, training_examples=len(train),
                  training_rows_sha256=digest(rows), validation_rows_sha256=digest(valid),
                  initial_adapter_sha256=adapter_digest(model), history=[],
                  initial_validation=validation(model, val, tok), optimizer_updates=0,
                  base=str(base), trainable_parameters=sum(p.numel() for p in params))
    output.mkdir(parents=True)
    opt = torch.optim.AdamW(params, lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    accum = settings['accumulation']
    total = math.ceil(len(train) / accum) * settings['epochs']
    best = float('inf')
    start = time.monotonic()
    for epoch in range(1, settings['epochs'] + 1):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(recipe['seed'] + epoch)).tolist()
        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.
        for index, key in enumerate(order):
            if index % accum == 0:
                warmup = max(1, math.ceil(total * settings['warmup_fraction']))
                scale = lr_multiplier(report['optimizer_updates'] + 1, total, warmup)
                for group in opt.param_groups:
                    group['lr'] = settings['learning_rate'] * scale
            loss, _ = objective(model, [train[key]], tok)
            group_size = min(accum, len(order) - (index // accum) * accum)
            (loss / group_size).backward()
            total_loss += float(loss.detach())
            if (index + 1) % accum == 0 or index + 1 == len(order):
                norm = torch.nn.utils.clip_grad_norm_(params, settings['gradient_clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite student gradient')
                opt.step()
                opt.zero_grad(set_to_none=True)
                report['optimizer_updates'] += 1
        scores = validation(model, val, tok)
        checkpoint = output / f'epoch_{epoch:02d}'
        model.save_pretrained(checkpoint)
        report['history'].append(dict(epoch=epoch, validation=scores, train_mean_ce=total_loss / len(train),
                                      optimizer_updates=report['optimizer_updates'], adapter_sha256=sha(checkpoint / 'adapter_model.safetensors')))
        if scores['token_ce'] < best:
            best = scores['token_ce']
            report.update(selected_epoch=epoch, selected_score=best)
        write_json(output / 'training.json', report)
        print(stage, epoch, scores, flush=True)
    report.update(status='complete', seconds=time.monotonic() - start)
    selected = output / f"epoch_{report['selected_epoch']:02d}"
    (output / 'selected').symlink_to(selected.name, target_is_directory=True)
    (output / 'final').symlink_to(f"epoch_{settings['epochs']:02d}", target_is_directory=True)
    write_json(output / 'training.json', report)
    del opt, params, model
    gc.collect()
    torch.cuda.empty_cache()
    return report


@torch.no_grad()
def generate(model, tok, cases, style, anchors=None):
    model.eval()
    answers = []
    for index, row in enumerate(cases):
        anchor = None if anchors is None else anchors[index]
        if anchors is not None and not anchor.strip():
            answers.append(dict(item_id=row['item_id'], reply='', status='failed_empty_anchor'))
            continue
        inputs = tok(source(row, style, anchor), return_tensors='pt')
        if inputs['input_ids'].shape[-1] > 512:
            raise ValueError('No generation truncation')
        seq = model.generate(**{k: v.cuda() for k, v in inputs.items()}, max_new_tokens=64,
                             num_beams=4, do_sample=False, no_repeat_ngram_size=3, use_cache=True)[0]
        reply = tok.decode(seq, skip_special_tokens=True).strip()
        status = 'failed_empty' if not reply else 'complete' if tok.eos_token_id in seq.tolist() else 'length_limit'
        answers.append(dict(item_id=row['item_id'], reply=reply, status=status, token_ids=seq.tolist()))
    return answers


def evaluate(base, tok, output):
    cases = read(ROOT / 'data/evaluation.json')
    fixed = read(ROOT / 'data/reference_evaluation.json')
    for checkpoint in ('selected', 'final'):
        model = load_switching_adapters(load_base(base), output / 'english' / checkpoint, output / 'singlish' / checkpoint)
        english = generate(model, tok, cases, 'STANDARD')
        activate_inference(model, 'singlish')
        anchors = [r['reply'] for r in english]
        singlish = generate(model, tok, cases, 'LOCAL_REWRITE', anchors)
        reference = generate(model, tok, fixed, 'LOCAL_REWRITE', [r['standard'] for r in fixed])
        activate_inference(model, 'english')
        # Actual adapter-switch regression on several diverse cases.
        if generate(model, tok, cases[:4], 'STANDARD') != english[:4]:
            raise ValueError('English output changed after adapter switch')
        write_json(output / f'evaluation_{checkpoint}.json', dict(
            pipeline=[dict(item_id=r['item_id'], english=e, standard_anchor=e['reply'], singlish=s)
                      for r, e, s in zip(cases, english, singlish, strict=True)],
            fixed_reference=[dict(item_id=r['item_id'], standard_anchor=r['standard'], singlish=s)
                             for r, s in zip(fixed, reference, strict=True)],
            english_switch_replay_passed=True, independent_test=False, quality_scored=False))
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['original', 'sealion', 'qwen'], default='original')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--scope', choices=['pilot', 'full'], default='full')
    args = parser.parse_args()
    torch.set_num_threads(4)
    recipe = read(ROOT / 'recipe.json')
    base = verify_base(recipe)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    manifest_path = ROOT / f'targets/{args.scope}/manifest.json'
    manifest = read(manifest_path)
    for name, expected in manifest['files'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Reviewed target snapshot changed: ' + name)
    arm_rows = {arm: read(ROOT / f'targets/{args.scope}/{arm}.json') for arm in ('original', 'sealion', 'qwen')}
    ids = [[r['item_id'] for r in rows] for rows in arm_rows.values()]
    if not ids[0] or not (ids[0] == ids[1] == ids[2]):
        raise ValueError('Students must see the same accepted input subset and order')
    if args.preflight:
        preflight(recipe, base, tok, arm_rows['sealion'], ROOT / f'students/preflight_{args.scope}.json')
        return
    if read(ROOT / f'students/preflight_{args.scope}.json')['status'] != 'passed':
        raise ValueError('Student fitting preflight must pass')
    output = ROOT / f'students/{args.scope}/{args.arm}'
    english = fit(recipe, base, tok, arm_rows[args.arm], read(ROOT / 'data/old_validation.json'), 'english', output / 'english')
    singlish = fit(recipe, base, tok, arm_rows[args.arm], read(ROOT / 'data/old_rewrite_validation.json'), 'singlish', output / 'singlish')
    if english['initial_adapter_sha256'] != singlish['initial_adapter_sha256']:
        raise ValueError('Independent adapters must have identical initial weights')
    evaluate(base, tok, output)
    verify_base(recipe)
    write_json(output / 'complete.json', dict(status='complete', arm=args.arm, scope=args.scope,
        common_conversations=len(arm_rows[args.arm]), shared_base_unchanged=True,
        initial_adapter_sha256=english['initial_adapter_sha256'], target_manifest_sha256=sha(manifest_path)))


if __name__ == '__main__':
    main()
