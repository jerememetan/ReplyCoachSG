"""Compare three fixed generators on validation using the existing Gemma rubric."""
import argparse
import ast
from datetime import datetime, timezone
import gc
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock
from replycoach.evaluation import load_evaluation_inputs
from replycoach.fallback import maybe_fallback
from replycoach.generate.providers import load_generator
from replycoach.io import ROOT, read_jsonl
from replycoach.judging import write_judging_packet
from replycoach.pipeline import prepare_input
from replycoach.systems import _generate
from replycoach.transformer_comparison import build_packet, summarise_comparison, task_key

MODELS = {'Original mT5': 'v5', 'New mT5 A': 'mt5_handoff', 'SEA-LION 3B': 'sealion_handoff'}
PRIOR = ROOT / 'experiments/judging/four_systems_20261026_01/main_packet'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    value = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def load_judge():
    spec = importlib.util.spec_from_file_location('existing_judge', ROOT / 'scripts/judge_replies.py')
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    return judge


def reuse_prior(judge, identity, personas, rubric):
    """Reuse only byte-equivalent prompts from the earlier validation evaluation."""
    cache = {}
    if not (PRIOR / 'ratings/run.json').exists():
        return cache
    receipt = read(PRIOR / 'ratings/run.json')
    checks = {**identity, 'personas': personas, 'rubric': rubric, 'response_schema': judge.SCHEMA,
              'calibration': False, 'provider': 'ollama'}
    if any(receipt.get(k) != v for k, v in checks.items()):
        print('Prior validation evaluator differs; its ratings will not be reused.', flush=True)
        return cache
    old_source = subprocess.check_output(['git', 'show', '96f09fb:scripts/judge_replies.py'], cwd=ROOT)
    if sha256(old_source).hexdigest() != receipt['script_sha256']:
        return cache
    def function_tree(source):
        module = ast.parse(source)
        node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'judge_one')
        return ast.dump(node, include_attributes=False)
    if function_tree(old_source) != function_tree((ROOT / 'scripts/judge_replies.py').read_bytes()):
        return cache
    public, mapping, hashes = judge.load_packet(PRIOR)
    if hashes != receipt['packet_hashes']:
        raise ValueError('Prior validation packet changed')
    by_id = {r['candidate_id']: r for r in public}
    run_id = sha256(json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    for row in read_jsonl(PRIOR / 'ratings/ratings.jsonl'):
        if row['run_id'] != run_id or row['candidate_id'] not in mapping:
            raise ValueError('Prior validation ratings do not match their receipt')
        if row['status'] == 'ok':
            judge.validate_rating(row['rating'])
            key = task_key(by_id[row['candidate_id']], row['persona_id'])
            cache.setdefault(key, (row['rating'], {'kind': 'prior_validation',
                'candidate_id': row['candidate_id'], 'run_id': run_id,
                'path': str(PRIOR.relative_to(ROOT))}))
    return cache


def generate_outputs(out, items):
    import torch
    torch.set_num_threads(4)
    # Free any previous judge before loading a generator on the same laptop GPU.
    judge = load_judge()
    config = judge.judge_settings(judge.read_env(ROOT / '.env'))
    judge.request_json(config['base'], '/api/generate', {'model': config['model'], 'keep_alive': 0})
    outputs = {}
    for model, provider in MODELS.items():
        path = out / f'{provider}_outputs.json'
        rows = read(path) if path.exists() else {}
        if not set(rows) <= {i.id for i in items}:
            raise ValueError('Unexpected saved validation outputs')
        generator = load_generator(provider) if len(rows) < len(items) else None
        try:
            for item in items:
                if item.id in rows:
                    continue
                prepared = prepare_input(item.message.text, item.memory_texts(), 'M2')
                started = time.monotonic()
                error = None
                try:
                    candidates = _generate(prepared, lambda: generator)
                except (ValueError, RuntimeError, OSError) as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    candidates = [maybe_fallback(None, item.message.text, prepared.memory, style)
                                  for style in ['STANDARD', 'LOCAL']]
                by_style = {c.style: c for c in candidates}
                rows[item.id] = dict(standard=by_style['STANDARD'].candidate_text,
                    local=by_style['LOCAL'].candidate_text, error=error,
                    candidates=[c.to_dict() for c in candidates], memory=prepared.memory,
                    seconds=round(time.monotonic()-started, 3))
                save(path, rows)
                save(out / 'progress.json', dict(phase='generating', model=model,
                    completed=len(rows), total=len(items), generation_failures=sum(bool(r['error']) for r in rows.values())))
                print(f'{model}: {len(rows)}/{len(items)} generated; failure={bool(error)}', flush=True)
        finally:
            if callable(getattr(generator, 'close', None)):
                generator.close()
            generator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        outputs[model] = rows
    return outputs


def write_report(out, report):
    lines = ['# Transformer validation comparison', '',
             '50 validation conversations, two styles and ten fictional Gemma personas. '
             'No held-out test cases or new grading criteria.', '',
             '| Model | Style | Appropriateness | Naturalness | Style fit | Would send |',
             '|---|---|---:|---:|---:|---:|']
    for row in report['summary']:
        lines.append(f"| {row['model']} | {row['style']} | {row['appropriateness']:.3f} | "
                     f"{row['naturalness']:.3f} | {row['style_fit']:.3f} | {100*row['would_send']:.1f}% |")
    lines += ['', f"Highest overall appropriateness point estimate: **{report['highest_point_estimate']}**.", '']
    if report['recommended_model']:
        lines.append(f"Recommended candidate on this validation comparison: **{report['recommended_model']}**.")
    else:
        lines.append('No clear winner: paired conversation confidence intervals include zero for at least one competitor.')
    for row in report['top_model_paired_comparisons']:
        low, high = row['appropriateness_delta_ci95']
        lines.append(f"- {row['top_model']} minus {row['other_model']}: "
                     f"{row['appropriateness_delta']:+.3f}; paired 95% bootstrap interval [{low:+.3f}, {high:+.3f}].")
    lines += ['', 'Generation failures (the existing safety fallback was judged and each failure remains recorded):']
    lines.extend(f'- {model}: {count}/50 conversations.' for model, count in report['generation_failures'].items())
    lines += ['', report['limitation'], '', 'This report does not automatically replace the production generator.']
    (out / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run(out, resume):
    if any(os.environ.get(k) for k in ['V5_CHECKPOINT', 'V5_LOCAL_REWRITE_CHECKPOINT',
                                      'REPLYCOACH_MT5_A_BUNDLE', 'REPLYCOACH_SEALION_BUNDLE']):
        raise ValueError('Unset model overrides for this fixed three-model comparison')
    inputs = load_evaluation_inputs(ROOT / 'data/dev/validation.jsonl', ROOT / 'data/dev/train.jsonl')
    if len(inputs.queries.items) != 50 or len(inputs.bank.items) != 190:
        raise ValueError('Expected the reviewed 50-query / 190-bank validation split')
    items = sorted(inputs.queries.items, key=lambda i: i.id)
    judge = load_judge()
    config = judge.judge_settings(judge.read_env(ROOT / '.env'))
    identity = judge.prepare_judge(config)
    personas = judge.load_personas(ROOT / 'data/evaluation/personas.csv')
    if len(personas) != 10:
        raise ValueError('Expected ten existing personas')
    rubric = judge.RUBRIC_PATH.read_text(encoding='utf-8')
    files = [Path(__file__), ROOT / 'scripts/judge_replies.py', ROOT / 'data/dev/validation.jsonl',
             ROOT / 'data/dev/train.jsonl', ROOT / 'data/evaluation/personas.csv', judge.RUBRIC_PATH,
             ROOT / 'src/replycoach/transformer_comparison.py', ROOT / 'src/replycoach/systems.py',
             ROOT / 'src/replycoach/memory.py', ROOT / 'src/replycoach/pipeline.py', ROOT / 'src/replycoach/fallback.py']
    files += sorted((ROOT / 'src/replycoach/generate').rglob('*.py'))
    for folder in ['artifacts/spk1_ul025_s425/best', 'artifacts/mt5_base_after_spans_spk/best']:
        files += sorted(p for p in (ROOT / folder).rglob('*') if p.is_file())
    for name in ['replycoach_mt5_A_v1', 'replycoach_sealion_repair25_v1']:
        files.append(ROOT / 'artifacts/model_handoff_v1' / name / 'manifest.json')
    freeze = dict(record_type='transformer_validation_selection', evaluation_split='validation', seed=425,
                  models=MODELS, memory='M2', comparison='generator outputs before reranking',
                  queries_fingerprint=inputs.queries.fingerprint, bank_fingerprint=inputs.bank.fingerprint,
                  judge=identity, personas=personas, rubric=rubric,
                  files={p.relative_to(ROOT).as_posix(): digest(p) for p in files},
                  selection='Equal-style mean appropriateness; 5000 paired conversation bootstrap samples',
                  failure_policy='Record decoding failure and grade the existing app safety fallback; never substitute another model')
    if resume:
        if read(out / 'freeze.json') != freeze:
            raise ValueError('Frozen run inputs changed; existing results were not overwritten')
    else:
        save(out / 'freeze.json', freeze)
    outputs = generate_outputs(out, items)
    raw_inputs = [dict(item_id=i.id, message=i.message.text,
                       conversation=[dict(speaker=t.speaker, text=t.text) for t in i.session_memory]) for i in items]
    public, private = build_packet(raw_inputs, outputs)
    packet = out / 'packet'
    if not packet.exists():
        write_judging_packet(packet, public, private)
    loaded, mapping, packet_hashes = judge.load_packet(packet)
    if loaded != public:
        raise ValueError('Saved blinded packet differs from generated outputs')
    receipt = dict(**identity, personas=personas, rubric=rubric, response_schema=judge.SCHEMA,
                   calibration=False, provider='ollama', packet_hashes=packet_hashes,
                   freeze_sha256=digest(out / 'freeze.json'))
    if (out / 'judge_run.json').exists() and read(out / 'judge_run.json') != receipt:
        raise ValueError('Judge run changed')
    save(out / 'judge_run.json', receipt)
    run_id = sha256(json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache = reuse_prior(judge, identity, personas, rubric)
    public_by_id = {r['candidate_id']: r for r in public}
    latest = {}
    journal = out / 'ratings.jsonl'
    for row in read_jsonl(journal) if journal.exists() else []:
        key = row['candidate_id'], row['persona_id']
        if row['run_id'] != run_id or key[0] not in public_by_id or key[1] not in {p['id'] for p in personas}:
            raise ValueError('Rating journal identity mismatch')
        if row['status'] == 'ok':
            judge.validate_rating(row['rating'])
            cache[task_key(public_by_id[key[0]], key[1])] = (row['rating'], {'kind': 'current_run', 'candidate_id': key[0]})
        latest[key] = row
    expected = len(public) * len(personas)
    with journal.open('a', encoding='utf-8') as handle:
        for candidate in public:
            for persona in personas:
                key = candidate['candidate_id'], persona['id']
                if latest.get(key, {}).get('status') == 'ok':
                    continue
                started = time.monotonic()
                row = dict(run_id=run_id, candidate_id=key[0], persona_id=key[1], status='error', rating=None,
                           attempts=latest.get(key, {}).get('attempts', 0),
                           created_at=datetime.now(timezone.utc).isoformat())
                cached = cache.get(task_key(candidate, persona['id']))
                if cached:
                    row.update(status='ok', rating=cached[0], provenance=cached[1])
                else:
                    for attempt in range(2):
                        row['attempts'] += 1
                        try:
                            rating, timing = judge.judge_one(config['base'], config['model'], persona, candidate, rubric)
                            row.update(status='ok', rating=rating, ollama=timing, provenance={'kind': 'new_gemma_call'})
                            cache[task_key(candidate, persona['id'])] = (rating, {'kind': 'current_run', 'candidate_id': key[0]})
                            break
                        except (OSError, ValueError, TypeError) as exc:
                            row['error'] = f'{type(exc).__name__}: {exc}'[:500]
                row['seconds'] = round(time.monotonic()-started, 3)
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                handle.flush(); os.fsync(handle.fileno())
                latest[key] = row
                done = sum(r['status']=='ok' for r in latest.values())
                new = sum(r.get('provenance', {}).get('kind')=='new_gemma_call' for r in latest.values())
                progress = dict(phase='judging', completed=done, total=expected, new_gemma_calls=new,
                                reused=done-new, errors=sum(r['status']!='ok' for r in latest.values()))
                save(out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    report = summarise_comparison(public, private, list(latest.values()), outputs, [p['id'] for p in personas])
    report.update(judge=identity, generation_latency={m: sum(r['seconds'] for r in rows.values())/len(rows)
                                                     for m, rows in outputs.items()},
                  rating_provenance={kind: sum(r.get('provenance', {}).get('kind')==kind for r in latest.values())
                                     for kind in ['new_gemma_call', 'prior_validation', 'current_run']})
    save(out / 'result.json', report)
    write_report(out, report)
    save(out / 'progress.json', dict(phase='completed', completed=expected, total=expected,
                                    highest_point_estimate=report['highest_point_estimate'],
                                    recommended_model=report['recommended_model']))
    print(json.dumps(report, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'experiments/judging'):
        raise ValueError('Use an ignored experiments/judging output folder')
    out.mkdir(parents=True, exist_ok=args.resume)
    with FileLock(str(out / '.lock'), timeout=0):
        try:
            run(out, args.resume)
        except Exception as exc:
            save(out / 'progress.json', dict(phase='failed', error=f'{type(exc).__name__}: {exc}'))
            raise


if __name__ == '__main__':
    main()
