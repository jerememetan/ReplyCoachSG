"""Continue the frozen 3,000-cell comparison on an allocated cluster GPU."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import time

from replycoach.io import ROOT, read_jsonl
from replycoach.transformer_comparison import summarise_comparison, task_key


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    value = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    out = ROOT / 'run'
    handoff = read(ROOT / 'handoff_manifest.json')
    for name, checksum in handoff['files'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError('Cluster handoff checksum mismatch: '+name)
    spec = importlib.util.spec_from_file_location('existing_judge', ROOT / 'scripts/judge_replies.py')
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    receipt = read(out / 'judge_run.json')
    config = judge.judge_settings({})
    identity = judge.prepare_judge(config)
    if any(receipt.get(k) != v for k, v in identity.items()):
        raise ValueError('Cluster Gemma model digest or settings differ from the laptop run')
    frozen = read(out / 'freeze.json')
    for name in ['scripts/judge_replies.py', 'src/replycoach/transformer_comparison.py']:
        if digest(ROOT / name) != frozen['files'][name]:
            raise ValueError('Judging/analysis source differs from the frozen laptop run')
    if digest(out / 'freeze.json') != receipt['freeze_sha256']:
        raise ValueError('Original freeze changed')
    personas, rubric = receipt['personas'], receipt['rubric']
    if (judge.load_personas(ROOT / 'data/evaluation/personas.csv') != personas
            or judge.RUBRIC_PATH.read_text(encoding='utf-8') != rubric
            or judge.SCHEMA != receipt['response_schema']):
        raise ValueError('Cluster rubric, personas or schema differ')
    public, mapping, hashes = judge.load_packet(out / 'packet')
    if hashes != receipt['packet_hashes']:
        raise ValueError('Frozen judging packet changed')
    private = list(mapping.values())
    by_id = {r['candidate_id']: r for r in public}
    expected_keys = {(r['candidate_id'], p['id']) for r in public for p in personas}
    run_id = sha256(json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    journal = out / 'ratings.jsonl'
    if not journal.exists():
        journal.write_bytes((out / 'ratings_seed.jsonl').read_bytes())
    latest, cache = {}, {}
    for row in read_jsonl(journal):
        key = row['candidate_id'], row['persona_id']
        if row['run_id'] != run_id or key not in expected_keys:
            raise ValueError('Saved rating identity mismatch')
        if latest.get(key, {}).get('status') == 'ok':
            raise ValueError('Duplicate completed cluster rating')
        if row['status'] == 'ok':
            judge.validate_rating(row['rating'])
            cache[task_key(by_id[key[0]], key[1])] = (row['rating'], key[0])
        latest[key] = row
    seeded = sum(r['status'] == 'ok' for r in latest.values())
    started, calls, reused = time.monotonic(), 0, 0
    first_scored = None
    runtime = dict(job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   ollama_version=judge.request_json(config['base'], '/api/version'),
                   original_run_id=run_id, seeded_ratings=seeded,
                   started_at=datetime.now(timezone.utc).isoformat())
    save(out / 'cluster_runtime.json', runtime)
    with journal.open('a', encoding='utf-8') as handle:
        for candidate in public:
            for persona in personas:
                key = candidate['candidate_id'], persona['id']
                if latest.get(key, {}).get('status') == 'ok':
                    continue
                row = dict(run_id=run_id, candidate_id=key[0], persona_id=key[1], status='error', rating=None,
                           attempts=latest.get(key, {}).get('attempts', 0), execution='cluster',
                           job_id=runtime['job_id'], node=runtime['node'],
                           created_at=datetime.now(timezone.utc).isoformat())
                tick = time.monotonic()
                cached = cache.get(task_key(candidate, persona['id']))
                if cached:
                    row.update(status='ok', rating=cached[0], provenance={'kind': 'current_run', 'candidate_id': cached[1]})
                    reused += 1
                else:
                    for attempt in range(2):
                        row['attempts'] += 1
                        calls += 1
                        try:
                            rating, timing = judge.judge_one(config['base'], config['model'], persona, candidate, rubric)
                            row.update(status='ok', rating=rating, ollama=timing, provenance={'kind': 'new_gemma_call'})
                            cache[task_key(candidate, persona['id'])] = (rating, key[0])
                            break
                        except (OSError, ValueError, TypeError) as exc:
                            row['error'] = f'{type(exc).__name__}: {exc}'[:500]
                row['seconds'] = round(time.monotonic()-tick, 3)
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                handle.flush(); os.fsync(handle.fileno())
                latest[key] = row
                if row['status'] == 'ok' and first_scored is None:
                    first_scored = time.monotonic()
                    runtime['model_residency'] = judge.request_json(config['base'], '/api/ps')
                    if not any(m.get('size_vram', 0) > 0 for m in runtime['model_residency'].get('models', [])):
                        raise RuntimeError('Ollama did not place the judge on a GPU')
                    save(out / 'cluster_runtime.json', runtime)
                progress = dict(phase='judging', completed=sum(r['status']=='ok' for r in latest.values()),
                    total=len(expected_keys), seed_ratings=seeded, cluster_calls=calls, cluster_reused=reused,
                    errors=sum(r['status']!='ok' for r in latest.values()),
                    elapsed_seconds=round(time.monotonic()-started, 2),
                    scoring_elapsed_seconds=round(time.monotonic()-first_scored, 2) if first_scored else 0,
                    last_call_seconds=row['seconds'], job_id=runtime['job_id'], node=runtime['node'])
                save(out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    outputs = {label: read(out / filename) for label, filename in {
        'Original mT5': 'v5_outputs.json', 'New mT5 A': 'mt5_handoff_outputs.json',
        'SEA-LION 3B': 'sealion_handoff_outputs.json'}.items()}
    report = summarise_comparison(public, private, list(latest.values()), outputs, [p['id'] for p in personas])
    report.update(judge=identity, cluster_runtime=runtime,
                  generation_latency={m: sum(r['seconds'] for r in rows.values())/len(rows) for m, rows in outputs.items()})
    save(out / 'result.json', report)
    lines = ['# Transformer validation comparison', '', report['primary_criterion'], '',
             '| Model | Style | Appropriateness | Naturalness | Style fit | Would send |',
             '|---|---|---:|---:|---:|---:|']
    for row in report['summary']:
        lines.append(f"| {row['model']} | {row['style']} | {row['appropriateness']:.3f} | "
                     f"{row['naturalness']:.3f} | {row['style_fit']:.3f} | {100*row['would_send']:.1f}% |")
    lines += ['', 'Highest point estimate: '+report['highest_point_estimate'],
              'Recommendation: '+str(report['recommended_model'] or 'No clear winner from paired intervals'),
              '', json.dumps(report['top_model_paired_comparisons'], indent=2), '',
              'Generation failures: '+json.dumps(report['generation_failures']), '', report['limitation']]
    (out / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save(out / 'progress.json', dict(phase='completed', completed=len(expected_keys), total=len(expected_keys),
                                    recommended_model=report['recommended_model'], job_id=runtime['job_id']))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(ROOT / 'run/progress.json', dict(phase='failed', error=f'{type(exc).__name__}: {exc}'))
        raise
