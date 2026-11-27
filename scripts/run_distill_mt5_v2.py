"""Generate train-only teacher English, then reviewed-English-conditioned Singlish."""
import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file

import run_decoder_compare_v1 as decoder
from replycoach.generate.decoder_compare_v1 import write_json
from replycoach.generate.distill_mt5_v2 import digest, input_only, reviewed

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def verify(recipe):
    for name, expected in recipe['payload_manifest'].items():
        if decoder.sha(ROOT / name) != expected:
            raise ValueError('Frozen payload changed: ' + name)


def teacher(recipe, arm, stage):
    spec = recipe['teachers'][arm][stage]
    path = Path(spec['path']).resolve()
    if decoder.sha(path / 'adapter_model.safetensors') != spec['sha256']:
        raise ValueError('Teacher adapter changed')
    model = PeftModel.from_pretrained(decoder.base_model(recipe, arm), path, is_trainable=False).eval()
    expected = load_file(str(path / 'adapter_model.safetensors'))
    actual = get_peft_model_state_dict(model)
    if set(expected) != set(actual) or any(not torch.equal(v, actual[k].detach().cpu()) for k, v in expected.items()):
        raise ValueError('Teacher adapter tensors failed strict replay')
    return model


def generate(recipe, arm, stage, scope):
    inputs = read(ROOT / 'data/train_inputs.json')
    if scope == 'pilot':
        wanted = set(recipe['pilot_ids'])
        inputs = [r for r in inputs if r['item_id'] in wanted]
    approved = None
    if stage == 'singlish':
        # Review is completed BEFORE the Singlish teacher sees an English anchor.
        raw = read(ROOT / f'outputs/{arm}/english_{scope}.json')['outputs']
        approved = reviewed(raw, read(ROOT / f'reviews/{arm}/english_{scope}.json')['decisions'], 'english')
        inputs = [r for r in inputs if r['item_id'] in approved]
    output = ROOT / f'outputs/{arm}/{stage}_{scope}.json'
    identity = dict(teacher=arm, stage=stage, scope=scope,
                    checkpoint=recipe['teachers'][arm][stage],
                    input_sha256=digest(inputs),
                    anchors_sha256=digest(approved),
                    recipe_sha256=decoder.sha(ROOT / 'recipe.json'))
    report = read(output) if output.exists() else dict(identity=identity, status='running', outputs=[])
    if report['identity'] != identity:
        raise ValueError('Resume identity changed')
    completed = {r['item_id']: r for r in report['outputs']}
    if not completed.keys() <= {r['item_id'] for r in inputs}:
        raise ValueError('Resume has unexpected inputs')
    if report['status'] == 'complete':
        print('Already complete', str(output), flush=True)
        return
    tok = decoder.tokenizer(recipe, arm)
    model = teacher(recipe, arm, stage)
    started = time.monotonic()
    for row in inputs:
        if row['item_id'] in completed:
            continue
        anchor = None if approved is None else approved[row['item_id']]['target']
        result = decoder.decode(model, tok, row, 'STANDARD' if stage == 'english' else 'LOCAL_REWRITE', anchor)
        item = dict(**input_only(row), **{stage: result})
        if anchor is not None:
            item['standard_anchor'] = anchor
        report['outputs'].append(item)
        report.update(completed=len(report['outputs']), requested=len(inputs), seconds_this_attempt=time.monotonic() - started)
        write_json(output, report)
        print(arm, stage, scope, len(report['outputs']), '/', len(inputs), row['item_id'], result['status'], flush=True)
    report['status'] = 'complete'
    write_json(output, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', choices=['sealion', 'qwen'], required=True)
    parser.add_argument('--stage', choices=['english', 'singlish'], required=True)
    parser.add_argument('--scope', choices=['pilot', 'full'], default='pilot')
    args = parser.parse_args()
    torch.set_num_threads(4)
    recipe = read(ROOT / 'recipe.json')
    verify(recipe)
    generate(recipe, args.teacher, args.stage, args.scope)


if __name__ == '__main__':
    main()
