"""Five reused development cases: raw-base prompts versus exported repair25.

Inference only. No demonstrations, target replies, updates or test-set access.
"""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import re
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def conversation(row):
    labels = {'user': 'You', 'peer': 'Friend', 'other': 'Other'}
    return '\n'.join(labels[speaker] + ': ' + text
                     for speaker, text in zip(row['speakers'], row['context'], strict=True))


def instruction_english(row):
    return (
        'Write one short reply to the incoming message as You.\n'
        "Speak from You's perspective using the conversation below.\n"
        'Keep known facts, people, locations and times consistent.\n'
        'Answer the incoming message directly. Do not invent extra conversation turns.\n'
        'Use standard English. Output only the reply.\n\n'
        'Conversation so far:\n' + conversation(row) + '\n'
        'Friend: ' + row['message'] + '\nYou (STANDARD):')


def instruction_rewrite(row, anchor):
    return (
        'Rewrite the English draft as one short, natural Singlish message from You.\n'
        'Preserve the same people, facts, time, location and decision.\n'
        'Output only the rewritten reply. Do not add dialogue turns.\n\n'
        'Conversation so far:\n' + conversation(row) + '\n'
        'Friend: ' + row['message'] + '\n'
        'English draft: ' + anchor.strip() + '\nYou (LOCAL_REWRITE):')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-rule', choices=['original', 'shared'], default='original')
    args = parser.parse_args()
    assert not args.output.exists(), 'Fresh output path required'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('portable_sealion', args.bundle / 'infer.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    import torch
    from replycoach.generate.sealion_data import source
    from replycoach.generate.sealion_dual_v1 import BOUNDARY
    from replycoach.generate.sealion_rewrite_v1 import rewrite_source, BOUNDARY as REWRITE_BOUNDARY
    from replycoach.generate.sealion_beam_v1 import beam_decode

    torch.set_num_threads(4)
    torch.manual_seed(425)
    torch.cuda.manual_seed_all(425)
    rows = json.loads(args.inputs.read_text())
    loader.validate_inputs(rows)
    assert len(rows) == 5
    pipeline = loader.ReplyPipeline(args.bundle, device='cuda', verify=True)
    assert pipeline.kind == 'sealion'
    assert pipeline.manifest['files']['base/model.safetensors']['sha256'] == 'acba6d6560f61ac72cc95feca5c89f6aabbbf1e9e89c18440237331fcb5ae107'
    model = pipeline.model
    version_snapshot = [(name, id(p), p._version) for name, p in model.named_parameters()]
    assert not any(p.requires_grad for p in model.parameters())
    layers = [m for m in model.modules() if hasattr(m, 'lora_A') and hasattr(m, 'disable_adapters')]
    assert layers
    english_boundary = (re.compile(r'\n(?:Friend|Other|You)(?: \([A-Z_]+\))?:')
                        if args.stop_rule == 'shared' else BOUNDARY)
    rewrite_boundary = (re.compile(r'\n(?:(?:Friend|Other|You)(?: \([A-Z_]+\))?|English draft):')
                        if args.stop_rule == 'shared' else REWRITE_BOUNDARY)
    report = dict(
        status='running', experiment='sealion_zero_shot_v2' if args.stop_rule == 'shared' else 'sealion_zero_shot_v1', inference_only=True,
        training_performed=False, base_identity=pipeline.manifest['model'],
        bundle=str(args.bundle), gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__, inputs_sha256=loader.sha(args.inputs),
        script_sha256=loader.sha(Path(__file__)), decoding=dict(num_beams=4,
            no_repeat_ngram_size=3, max_new_tokens=64, length_penalty=1.),
        cases=rows, conditions={}, prompts={},
        scope='Five previously inspected development cases. Prompt variants are diagnostics; no human accuracy score or final-model change.',
        stop_rule=args.stop_rule, english_boundary=english_boundary.pattern,
        rewrite_boundary=rewrite_boundary.pattern,
        note='Base arms use the exact packaged pretrained base with every LoRA layer disabled. Both stages use no adapters. Instruction arm has no labelled demonstrations.',
        strict_loading=pipeline.loading_problems)
    save(args.output, report)
    started = time.monotonic()

    @torch.no_grad()
    def generate(prompt, boundary):
        ids = pipeline.tokenizer.encode(prompt, add_special_tokens=False)
        return beam_decode(model, pipeline.tokenizer, ids, boundary, num_beams=4,
                           no_repeat_ngram_size=3, max_new_tokens=64,
                           length_penalty=1., device='cuda')

    conditions = ['base_matched_prompt', 'base_explicit_instruction']
    if args.stop_rule == 'shared':
        conditions.append('repair25_shared_stop')
    for condition in conditions:
        untouched = condition.startswith('base_')
        report['conditions'][condition] = []
        report['prompts'][condition] = []
        with model.disable_adapter() if untouched else contextlib.nullcontext():
            if untouched:
                assert all(layer.disable_adapters for layer in layers)
            model.eval()
            for row in rows:
                if not untouched:
                    pipeline._activate('english')
                explicit = condition == 'base_explicit_instruction'
                english_prompt = instruction_english(row) if explicit else source(row, 'STANDARD')
                english = generate(english_prompt, english_boundary)
                local_prompt = None
                if english['reply'].strip():
                    local_prompt = (instruction_rewrite(row, english['reply']) if explicit
                                    else rewrite_source(row, english['reply']))
                    if not untouched:
                        pipeline._activate('singlish')
                    local = generate(local_prompt, rewrite_boundary)
                else:
                    local = dict(reply='', status='failed_empty_anchor')
                report['conditions'][condition].append(dict(item_id=row['item_id'],
                    english=english['reply'], singlish=local['reply'], standard_anchor=english['reply'],
                    english_status=english['status'], singlish_status=local['status'],
                    english_details=english, singlish_details=local))
                report['prompts'][condition].append(dict(item_id=row['item_id'],
                    english=english_prompt, singlish=local_prompt))
                save(args.output, report)
                print(condition, row['item_id'], english['status'], local['status'], flush=True)

    report['conditions']['repair25'] = pipeline.generate_pairs(rows)
    expected = json.loads(args.expected.read_text())
    report['repair25_replay_differences'] = [dict(expected=e, actual=a)
        for a, e in zip(report['conditions']['repair25'], expected, strict=True)
        if any(a[k] != e[k] for k in ['item_id', 'english', 'singlish'])]
    if args.stop_rule == 'shared':
        report['repair25_stop_rule_differences'] = [dict(original=o, shared=s)
            for o, s in zip(report['conditions']['repair25'], report['conditions']['repair25_shared_stop'], strict=True)
            if any(o[k] != s[k] for k in ['item_id', 'english', 'singlish'])]
    assert version_snapshot == [(name, id(p), p._version) for name, p in model.named_parameters()]
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    report.update(status='complete', model_parameters_unchanged=True,
        seconds=time.monotonic()-started,
        flags=[dict(condition=condition, item_id=row['item_id'],
                    english_status=row['english_status'], singlish_status=row['singlish_status'])
               for condition, outputs in report['conditions'].items() for row in outputs
               if row['english_status'] != 'complete' or row['singlish_status'] != 'complete'])
    save(args.output, report)
    print(json.dumps(dict(status=report['status'], cases=len(rows),
        conditions=list(report['conditions']), replay_differences=len(report['repair25_replay_differences']),
        flags=report['flags'], seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
