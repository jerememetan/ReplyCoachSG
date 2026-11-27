"""Independently check a completed matched judge snapshot before publishing it."""
from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/mt5_distill_judge_v1/stage'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    source = snapshot / 'results'
    verification = read(snapshot / 'verification.json')
    for name, info in verification['files'].items():
        assert sha(snapshot / name) == info['sha256'], name
    for name, expected in read(STAGE / 'payload_manifest.json').items():
        assert sha(STAGE / name) == expected, name
    complete = read(source / 'complete.json')
    assert complete['status'] == 'complete' and complete['all_valid_complete']
    assert complete['unique_judgments'] == 3390
    assert complete['expanded_ratings'] == 5000
    assert complete['conversations'] == 50 and complete['personas'] == 10
    assert complete['test_set_used'] is False
    assert '373729|COMPLETED|0:0|' in verification['accounting']
    config = read(STAGE / 'config.json')
    public = read(STAGE / 'judge_inputs.json')
    mapping = read(STAGE / 'PRIVATE_mapping.json')
    runner = module('matched_judge_runner', ROOT / 'scripts/run_mt5_distill_judge_v1.py')
    sys.path.insert(0, str(STAGE / 'src'))
    judge = module('frozen_groupmate_judge', STAGE / 'scripts/groupmate_judge.py')
    runner.verify_packet(config, public, mapping)
    identity = read(source / 'run.json')
    run_id = runner.stable_sha(identity)
    assert run_id == complete['run_id']
    assert identity['config_sha256'] == sha(STAGE / 'config.json')
    assert identity['payload_manifest_sha256'] == sha(STAGE / 'payload_manifest.json')
    assert identity['model_digest'] == config['expected_model_digest']
    tasks = runner.tasks_for(public, config['personas'])
    expected = {(r['candidate_id'], p['id']) for r, p in tasks}
    latest = runner.load_journal(source / 'ratings.jsonl', run_id, expected, judge)
    assert set(latest) == expected
    assert all(r['status'] == 'ok' and r['error'] is None for r in latest.values())
    assert max(r['ollama']['prompt_eval_count'] for r in latest.values()) < config['options']['num_ctx']
    expanded, summaries, paired = runner.expand_and_summarise(config, public, mapping, latest)
    assert summaries == read(source / 'summary.json')
    assert paired == read(source / 'paired_differences.json')
    saved_expanded = [json.loads(line) for line in (source / 'expanded_ratings.jsonl').read_text().splitlines()]
    assert expanded == saved_expanded
    cells = Counter((r['model'], r['style'], r['item_id']) for r in expanded)
    assert len(cells) == 500 and set(cells.values()) == {10}
    for job in ['373725', '373729']:
        runtime = read(source / f'runtime_{job}.json')
        assert runtime['judge']['model_digest'] == config['expected_model_digest']
        assert runtime['judge']['options'] == config['options']
        assert runtime['judge']['think'] == config['think']
        assert runtime['loaded_models'] and all(m['size_vram'] > 0 for m in runtime['loaded_models'])
    out = ROOT / 'artifacts/mt5_distill_judge_v1/final_results_20261121'
    out.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in {'.json', '.jsonl', '.csv'}:
            shutil.copy2(path, out / path.name)
    spot_ids = {'ITEM-B-057', 'ITEM-A-024', 'ITEM-A-046', 'ITEM-D-038',
                'ITEM-E-015', 'ITEM-E-028', 'ITEM-C-045', 'ITEM-D-013'}
    spotchecks = []
    for row in public:
        meta = mapping[row['candidate_id']]
        if meta['item_id'] in spot_ids:
            spotchecks.append({**row, **meta,
                'ratings': [{'persona_id': p['id'], **latest[(row['candidate_id'], p['id'])]['rating']}
                            for p in config['personas']]})
    (out / 'spotcheck_packet.json').write_text(json.dumps(spotchecks, indent=2, ensure_ascii=False) + '\n')
    record = {
        'experiment': 'mt5_distill_judge_v1', 'status': 'complete',
        'snapshot': str(snapshot.relative_to(ROOT)), 'results_directory': str(out.relative_to(ROOT)),
        'observed_at_utc': verification['observed_at_utc'], 'accounting': verification['accounting'],
        'run_id': run_id, 'judge_model': config['judge_model'], 'model_digest': config['expected_model_digest'],
        'protocol': 'Unchanged groupmate judge_one, rubric, ten personas, temperature 0, seed 425, think false; four concurrent fresh-context requests.',
        'scope': config['scope'], 'unique_requests': len(latest), 'expanded_ratings': len(expanded),
        'coverage': '5 systems × 2 styles × 50 same conversations × 10 personas; complete.',
        'failed_latest_ratings': 0,
        'requests_with_retry': sum(r['attempts'] > 1 for r in latest.values()),
        'max_prompt_tokens': max(r['ollama']['prompt_eval_count'] for r in latest.values()),
        'all_ratings_schema_valid': True, 'gpu_runtime_verified': True,
        'collection_sha256_verified': True, 'aggregation_independently_recomputed': True,
        'aggregation': config['aggregation'], 'deduplication': config['deduplication'],
        'summary': summaries, 'paired_differences': paired,
        'limitations': [
            'Reused development validation cases; these are not untouched test results.',
            'Ten personas are simulated by one judge, not independent human participants.',
            'Judge assesses each draft separately and does not score English/Singlish meaning consistency.',
            'LLM judgments can miss role/timing contradictions and overrate generic replies.',
            'Bootstrap intervals reflect conversation variation, not all judge/model uncertainty.',
            'Checkpoint selection used validation loss before this run; comparison concerns selected checkpoints only.',
            'Historical judge aggregate scores should not be mixed into this matched run.'
        ],
        'report_document_modified': False, 'source_models_modified': False,
    }
    dest = ROOT / 'experiments/mt5_distill_judge_v1_results.json'
    dest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    hashes = {str(p.relative_to(out)): sha(p) for p in out.iterdir() if p.is_file() and p.name != 'collection_sha256.json'}
    (out / 'collection_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({'result': str(dest), 'all_valid_complete': True, 'summary': summaries}, indent=2))


if __name__ == '__main__':
    main()
