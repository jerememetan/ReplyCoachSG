"""Independent, gated DSKD-CMA experiment; preserve all CE control artifacts."""
import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import train_distill_mt5_v2 as ce
import run_decoder_compare_v1 as decoder
from run_distill_mt5_v2 import teacher
from replycoach.generate.decoder_compare_v1 import lr_multiplier, write_json
from replycoach.generate.distill_mt5_v2 import digest
from replycoach.generate.sealion_audited_restart_v1 import prompt
from replycoach.generate.mt5_soft_kd_v1 import (
    causal_response_positions, FrozenHead, DualSpaceCMA, mt5_decoder_head,
)

ROOT = Path(__file__).resolve().parents[1]
ce.ROOT = ROOT


def read(path):
    return json.loads(Path(path).read_text())


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def tensor_digest(tensors):
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cache_identity(recipe, arm, stage, rows):
    return dict(arm=arm, stage=stage, rows_sha256=digest(rows),
                teacher=recipe['teachers'][arm][stage],
                protocol_sha256=decoder.sha(ROOT / 'soft_recipe.json'),
                reviewed_targets_sha256=decoder.sha(ROOT / f'targets/full/{arm}.json'),
                target_encoding='Native teacher-forced accepted text, separately encoded after native prompt, plus EOS; not saved beam token IDs.')


@torch.no_grad()
def build_cache(recipe, arm, stage, rows):
    output = ROOT / f'cache/{arm}/{stage}'
    if output.exists():
        raise FileExistsError('Preserve prior cache attempt: ' + str(output))
    tok = decoder.tokenizer(recipe, arm)
    model = teacher(recipe, arm, stage).requires_grad_(False).eval()
    base = model.get_base_model()
    native_head = base.get_output_embeddings()
    head = FrozenHead(native_head.weight, getattr(native_head, 'bias', None),
                      post_scale=getattr(base, 'logit_scale', None) or 1.)
    if head.post_scale == 0:
        raise ValueError('Uniform teacher head is invalid')
    head_sha = tensor_digest(head.state_dict())
    records, replay_max = [], 0.
    tok_signature = hashlib.sha256(json.dumps(tok.get_vocab(), sort_keys=True).encode()).hexdigest()
    started = time.monotonic()
    for index, row in enumerate(rows):
        text = row['standard' if stage == 'english' else 'local']
        anchor = row['standard'] if stage == 'singlish' else None
        native_prompt = prompt(row, 'STANDARD' if stage == 'english' else 'LOCAL_REWRITE', anchor)
        pids = tok.encode(native_prompt, add_special_tokens=False)
        tids = tok.encode(text, add_special_tokens=False)
        if tok.decode(tids, skip_special_tokens=True).strip() != text.strip():
            raise ValueError('Teacher target fails text roundtrip: ' + row['item_id'])
        combined, target, positions = causal_response_positions(pids, tids, tok.eos_token_id)
        ids = torch.tensor([combined], device='cuda')
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                    output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1]
        # Verify exact native head mapping (including legacy scaling), not a
        # guessed lm_head attribute or invented common vocabulary.
        replay = head(hidden)
        torch.testing.assert_close(replay.float(), out.logits.float(), rtol=1e-5, atol=1e-5)
        replay_max = max(replay_max, float((replay.float() - out.logits.float()).abs().max()))
        previous = ids[:, positions]
        target_ids = torch.tensor([target], device='cuda')
        embed = base.get_input_embeddings()
        records.append(dict(item_id=row['item_id'], labels=target_ids.cpu(),
            hidden=hidden[:, positions].detach().cpu(),
            index=torch.cat((embed(previous), embed(target_ids)), -1).detach().cpu(),
            target_embed=embed(target_ids).detach().cpu(),
            prompt_sha256=hashlib.sha256(native_prompt.encode()).hexdigest(),
            prompt_tokens=len(pids), response_tokens=len(target)))
        del out, hidden, replay
        if (index + 1) % 100 == 0:
            print('cache', arm, stage, index + 1, '/', len(rows), flush=True)
    if tensor_digest(head.state_dict()) != head_sha or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen teacher changed or acquired gradients')
    output.mkdir(parents=True)
    head_file = output.parent / 'head.pt'
    head_record = dict(state={k: v.cpu() for k, v in head.state_dict().items()},
                       pre_scale=head.pre_scale, post_scale=head.post_scale, tensor_sha256=head_sha)
    if head_file.exists():
        old = torch.load(head_file, weights_only=True)
        if old['tensor_sha256'] != head_sha:
            raise ValueError('Teacher heads differ across stages')
    else:
        torch.save(head_record, head_file)
    torch.save(records, output / 'responses.pt')
    report = dict(status='complete', identity=cache_identity(recipe, arm, stage, rows),
                  examples=len(records), tokenizer_vocab_sha256=tok_signature,
                  head_sha256=decoder.sha(head_file), response_sha256=decoder.sha(output / 'responses.pt'),
                  head_tensor_sha256=head_sha, teacher_frozen=True, head_replay_max_abs=replay_max,
                  seconds=time.monotonic() - started,
                  max_prompt_tokens=max(x['prompt_tokens'] for x in records),
                  max_response_tokens=max(x['response_tokens'] for x in records))
    write_json(output / 'manifest.json', report)
    del model, base, native_head, head, embed, records, head_record
    clear()
    return report


def load_cache(recipe, arm, stage, rows):
    output = ROOT / f'cache/{arm}/{stage}'
    report = read(output / 'manifest.json')
    if report['status'] != 'complete' or report['identity'] != cache_identity(recipe, arm, stage, rows):
        raise ValueError('Cache identity mismatch')
    head_file = output.parent / 'head.pt'
    for path, key in ((head_file, 'head_sha256'), (output / 'responses.pt', 'response_sha256')):
        if decoder.sha(path) != report[key]:
            raise ValueError('Frozen teacher cache changed')
    records = torch.load(output / 'responses.pt', weights_only=True)
    if [x['item_id'] for x in records] != [x['item_id'] for x in rows]:
        raise ValueError('Cache row order mismatch')
    saved = torch.load(head_file, weights_only=True, map_location='cuda')
    if tensor_digest(saved['state']) != saved['tensor_sha256']:
        raise ValueError('Cached teacher head tensor mismatch')
    head = FrozenHead(saved['state']['weight'], saved['state'].get('bias'),
                      saved['pre_scale'], saved['post_scale'])
    return records, head, report


def student_head(model):
    # Verified pinned forward calls lm_head(decoder_outputs[0]) directly. Do
    # not add a scale inferred from the historical tie_word_embeddings flag.
    return mt5_decoder_head(model.get_base_model())


def alignment(model, teacher_width, protocol):
    # Projector initialization must not consume the student's dropout RNG stream.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(protocol['projector_seed'])
        torch.cuda.manual_seed_all(protocol['projector_seed'])
        return DualSpaceCMA(model.config.d_model, teacher_width, protocol['temperature'], protocol['alpha']).cuda()


def objective(model, kd, batch, cached, tok, sh, th, replay=False):
    x, a, y = ce.tensor([batch], tok)
    base = model.get_base_model()
    decoder_ids = base.prepare_decoder_input_ids_from_labels(labels=y)
    out = model(input_ids=x, attention_mask=a, labels=y, use_cache=False, output_hidden_states=True)
    hidden = out.decoder_hidden_states[-1]
    if replay:
        torch.testing.assert_close(sh(hidden).float(), out.logits.float(), rtol=1e-5, atol=1e-5)
    # mT5 encoder positions are not causal response positions. Shifted decoder
    # IDs and corresponding labels define prediction indices in its own space.
    embed = base.get_input_embeddings()
    with torch.no_grad():
        index = torch.cat((embed(decoder_ids), embed(y.masked_fill(y.eq(-100), tok.pad_token_id))), -1)
    values = {k: cached[k].cuda() for k in ('hidden', 'index', 'target_embed', 'labels')}
    return kd(out.logits, hidden, index, y, values['hidden'], values['index'],
              values['target_embed'], values['labels'], sh, th)


def assert_frozen(model, head):
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError('Student base acquired gradients')
    if any(p.requires_grad or p.grad is not None for p in head.parameters()):
        raise ValueError('Teacher head acquired gradients')


@torch.no_grad()
def diagnostics(model, kd, data, cached, tok, sh, th):
    model.eval()
    totals = {}
    for batch, entry in zip(data, cached, strict=True):
        _, metrics = objective(model, kd, batch, entry, tok, sh, th, replay=True)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + float(value) / len(data)
    return totals


def pilot(recipe, protocol, base, tok, arm, stage, rows):
    output = ROOT / f'pilot/{arm}_{stage}.json'
    if output.exists():
        raise FileExistsError('Preserve prior pilot evidence')
    records, th, cache_report = load_cache(recipe, arm, stage, rows)
    model, params = ce.new_model(base, recipe)
    initial = ce.adapter_digest(model)
    if initial != protocol['expected_initial_adapter_sha256']:
        raise ValueError('KD does not match CE fresh initialization')
    sh = student_head(model)
    head_initial = tensor_digest(sh.state_dict())
    kd = alignment(model, records[0]['hidden'].shape[-1], protocol)
    all_params = params + list(kd.parameters())
    data = ce.encoded(rows[:4], tok, stage, recipe['student'])
    before = diagnostics(model, kd, data, records[:4], tok, sh, th)
    opt = torch.optim.AdamW(all_params, lr=recipe['student']['learning_rate'], weight_decay=.01)
    lora_grad = projector_grad = False
    for step in range(protocol['pilot_updates']):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, metrics = objective(model, kd, data[step % 4], records[step % 4], tok, sh, th)
        loss.backward()
        lora_grad |= any(p.grad is not None and p.grad.norm() > 0 for p in params)
        projector_grad |= all(p.grad is not None and p.grad.norm() > 0 for p in kd.parameters())
        assert_frozen(model, th)
        norm = torch.nn.utils.clip_grad_norm_(all_params, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite pilot gradient')
        opt.step()
    after = diagnostics(model, kd, data, records[:4], tok, sh, th)
    unchanged = tensor_digest(sh.state_dict()) == head_initial
    passed = (after['total'] < before['total'] and lora_grad and projector_grad and unchanged
              and ce.adapter_digest(model) != initial)
    report = dict(status='passed' if passed else 'failed', arm=arm, stage=stage, before=before, after=after,
        initial_adapter_sha256=initial, updates=protocol['pilot_updates'],
        finite_gradients=True, nonzero_student_gradients=bool(lora_grad),
        nonzero_projector_gradients=bool(projector_grad), base_head_unchanged=unchanged,
        base_no_gradients=True, teacher_frozen=True, native_student_head_replay=True,
        native_teacher_head_replay=cache_report['head_replay_max_abs'],
        checkpoint_created=False, projector_parameters=sum(p.numel() for p in kd.parameters()))
    write_json(output, report)
    if not passed:
        raise ValueError('Actual checkpoint KD pilot failed')
    del opt, all_params, params, model, kd, sh, th, records
    clear()
    return report


def fit(recipe, protocol, base, tok, arm, stage, rows, valid, output):
    if output.exists():
        raise FileExistsError('Preserve previous KD fitting attempt: ' + str(output))
    settings = recipe['student']
    records, th, cache_report = load_cache(recipe, arm, stage, rows)
    model, params = ce.new_model(base, recipe)
    initial = ce.adapter_digest(model)
    if initial != protocol['expected_initial_adapter_sha256']:
        raise ValueError('KD initialization differs from CE')
    sh = student_head(model)
    kd = alignment(model, records[0]['hidden'].shape[-1], protocol)
    all_params = params + list(kd.parameters())
    train = ce.encoded(rows, tok, stage, settings)
    val = ce.encoded(valid, tok, stage, settings)
    report = dict(status='running', arm=arm, stage=stage, training_examples=len(train),
        training_rows_sha256=digest(rows), validation_rows_sha256=digest(valid),
        initial_adapter_sha256=initial, history=[], initial_validation=ce.validation(model, val, tok),
        optimizer_updates=0, base=str(base), student_parameters=sum(p.numel() for p in params),
        projector_parameters=sum(p.numel() for p in kd.parameters()),
        cache_manifest_sha256=decoder.sha(ROOT / f'cache/{arm}/{stage}/manifest.json'),
        protocol=protocol, checkpoint_inference='Student LoRA only; saved projectors are training evidence.')
    # Check the actual mT5 head before optimizing any full-run examples.
    diagnostics(model, kd, train[:1], records[:1], tok, sh, th)
    output.mkdir(parents=True)
    opt = torch.optim.AdamW(all_params, lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    accum = settings['accumulation']
    total = math.ceil(len(train) / accum) * settings['epochs']
    best, started = float('inf'), time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, settings['epochs'] + 1):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(recipe['seed'] + epoch)).tolist()
        model.train()
        opt.zero_grad(set_to_none=True)
        sums = {}
        for index, key in enumerate(order):
            if index % accum == 0:
                warmup = max(1, math.ceil(total * settings['warmup_fraction']))
                scale = lr_multiplier(report['optimizer_updates'] + 1, total, warmup)
                for group in opt.param_groups:
                    group['lr'] = settings['learning_rate'] * scale
            loss, metrics = objective(model, kd, train[key], records[key], tok, sh, th)
            group_size = min(accum, len(order) - (index // accum) * accum)
            (loss / group_size).backward()
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.) + float(value) / len(train)
            if (index + 1) % accum == 0 or index + 1 == len(order):
                assert_frozen(model, th)
                norm = torch.nn.utils.clip_grad_norm_(all_params, settings['gradient_clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite KD gradient')
                opt.step()
                opt.zero_grad(set_to_none=True)
                report['optimizer_updates'] += 1
        scores = ce.validation(model, val, tok)
        checkpoint = output / f'epoch_{epoch:02d}'
        model.save_pretrained(checkpoint)
        torch.save(kd.state_dict(), checkpoint / 'training_only_projectors.pt')
        report['history'].append(dict(epoch=epoch, validation=scores, train_metrics=sums,
            optimizer_updates=report['optimizer_updates'], adapter_sha256=decoder.sha(checkpoint / 'adapter_model.safetensors'),
            projector_sha256=decoder.sha(checkpoint / 'training_only_projectors.pt')))
        if scores['token_ce'] < best:
            best = scores['token_ce']
            report.update(selected_epoch=epoch, selected_score=best)
        write_json(output / 'training.json', report)
        print(arm, stage, epoch, scores, sums, flush=True)
    if report['optimizer_updates'] != total:
        raise ValueError('Optimizer update budget mismatch')
    report.update(status='complete', seconds=time.monotonic() - started,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    (output / 'selected').symlink_to(f"epoch_{report['selected_epoch']:02d}", target_is_directory=True)
    (output / 'final').symlink_to(f"epoch_{settings['epochs']:02d}", target_is_directory=True)
    write_json(output / 'training.json', report)
    del opt, all_params, params, model, kd, sh, th, records
    clear()
    return report


def verify_inputs():
    recipe = read(ROOT / 'recipe.json')
    protocol = read(ROOT / 'soft_recipe.json')
    payload = read(ROOT / 'soft_payload_manifest.json')
    for name, expected in payload.items():
        if decoder.sha(ROOT / name) != expected:
            raise ValueError('Soft KD frozen payload changed: ' + name)
    # This manifest is the existing matched target freeze; no refreeze or review edits.
    if decoder.sha(ROOT / 'targets/full/manifest.json') != protocol['target_manifest_sha256']:
        raise ValueError('Reviewed target manifest changed')
    manifest = read(ROOT / 'targets/full/manifest.json')
    for name, expected in manifest['files'].items():
        if decoder.sha(ROOT / name) != expected:
            raise ValueError('Reviewed matched target changed: ' + name)
    rows = {arm: read(ROOT / f'targets/full/{arm}.json') for arm in ('original', 'sealion', 'qwen')}
    ids = [[r['item_id'] for r in v] for v in rows.values()]
    if not (len(ids[0]) == 571 and ids[0] == ids[1] == ids[2]):
        raise ValueError('Expected same 571 matched inputs')
    for arm in ('sealion', 'qwen'):
        for stage in ('english', 'singlish'):
            path = Path(recipe['teachers'][arm][stage]['path']) / 'adapter_model.safetensors'
            if decoder.sha(path) != recipe['teachers'][arm][stage]['sha256']:
                raise ValueError('Preserved teacher adapter changed')
    return recipe, protocol, ce.verify_base(recipe), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['gate', 'fit'], required=True)
    parser.add_argument('--arm', choices=['sealion', 'qwen'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    recipe, protocol, base, rows = verify_inputs()
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if args.mode == 'gate':
        gate = ROOT / 'pilot/gate.json'
        if gate.exists():
            raise FileExistsError('Gate already exists')
        report = dict(status='running', protocol_sha256=decoder.sha(ROOT / 'soft_recipe.json'), stages={})
        for arm in ('sealion', 'qwen'):
            for stage in ('english', 'singlish'):
                if (ROOT / f'cache/{arm}/{stage}/manifest.json').exists():
                    cached, cached_head, cached_report = load_cache(recipe, arm, stage, rows[arm])
                    if cached_report['head_replay_max_abs'] != 0. or not cached_report['teacher_frozen']:
                        raise ValueError('Imported cache lacks exact teacher replay')
                    del cached, cached_head, cached_report
                    clear()
                else:
                    build_cache(recipe, arm, stage, rows[arm])
                report['stages'][f'{arm}_{stage}'] = pilot(recipe, protocol, base, tok, arm, stage, rows[arm])
                write_json(gate, report)
        ce.verify_base(recipe)
        report.update(status='passed', base_unchanged=True, timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write_json(gate, report)
        return
    if not args.arm:
        raise ValueError('--arm required for fitting')
    gate = read(ROOT / 'pilot/gate.json')
    if gate['status'] != 'passed' or gate['protocol_sha256'] != decoder.sha(ROOT / 'soft_recipe.json'):
        raise ValueError('All actual-checkpoint pilot stages must pass first')
    output = ROOT / f'students/full/{args.arm}'
    for stage, valid in (('english', 'old_validation'), ('singlish', 'old_rewrite_validation')):
        fit(recipe, protocol, base, tok, args.arm, stage, rows[args.arm], read(ROOT / f'data/{valid}.json'), output / stage)
    ce.evaluate(base, tok, output)
    ce.verify_base(recipe)
    write_json(output / 'complete.json', dict(status='complete', arm=args.arm,
        common_conversations=571, shared_base_unchanged=True,
        target_manifest_sha256=protocol['target_manifest_sha256'], inference_projectors_required=False))


if __name__ == '__main__':
    main()
