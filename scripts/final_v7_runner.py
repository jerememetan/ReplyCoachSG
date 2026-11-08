"""Supplement the completed frozen test with V7's final Local replies."""
from collections import defaultdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory

from filelock import FileLock

from replycoach.final_v7 import make_v7_packet
from replycoach.io import LEXICON_PATH, ROOT, read_jsonl
from replycoach.judging import write_judging_packet
from replycoach.lexicon_refinement import refine_local_reply
from replycoach.schema import Candidate


BASELINE = ROOT / 'experiments/judging/final_test_20261102_01'
SEED = 425


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _save(path, payload):
    staging = path.with_suffix('.tmp')
    staging.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    staging.replace(path)


def _digest(path):
    with path.open('rb') as handle:
        checksum = sha256()
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _load_ratings(folder, candidate_ids, manifest, judge):
    personas = {p['id'] for p in manifest['personas']}
    run_id = sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    ratings = {}
    for row in read_jsonl(folder / 'ratings.jsonl'):
        key = (row['candidate_id'], row['persona_id'])
        if (row['run_id'] != run_id or key[0] not in candidate_ids or key[1] not in personas
                or key in ratings or row['status'] not in ('ok', 'error')):
            raise ValueError('Rating journal does not match the frozen packet')
        if row['status'] == 'ok':
            ratings[key] = judge.validate_rating(row['rating'])
    if len(ratings) != len(candidate_ids) * len(personas):
        raise ValueError('Rating journal does not contain all successful judgments')
    return ratings


def _report(mapping, v6, v7, ratings, personas, metrics):
    groups = defaultdict(list)
    for (cid, _), rating in ratings.items():
        if cid in mapping:
            row = mapping[cid]
            groups[(row['system'], row['style'])].append(rating)
    for row in [*v6, *v7]:
        groups[(row['system'], row['style'])].extend(ratings[(row['candidate_id'], p)] for p in personas)
    summary = [dict(system=system, style=style, ratings=len(rows),
                    **{metric: sum(r[metric] for r in rows) / len(rows) for metric in metrics})
               for (system, style), rows in sorted(groups.items())]
    v6_local = {s['item_id']: s['candidate_id'] for s in v6 if s['style'] == 'LOCAL'}
    paired = {}
    for row in v7:
        if row['style'] == 'LOCAL':
            before, after = v6_local[row['item_id']], row['candidate_id']
            paired[row['item_id']] = {
                metric: sum(ratings[(after, p)][metric] - ratings[(before, p)][metric]
                            for p in personas) / len(personas) for metric in metrics}
    delta = {metric: sum(row[metric] for row in paired.values()) / len(paired) for metric in metrics}
    return summary, paired, delta


def run_extension(out: Path, *, resume: bool):
    if out == BASELINE.resolve() or out.is_relative_to(BASELINE.resolve()):
        raise ValueError('Choose an output folder separate from the completed baseline')
    if resume != out.is_dir():
        raise ValueError('Use --resume only for an existing V7 run folder')
    out.mkdir(parents=True, exist_ok=resume)
    with FileLock(str(out / '.lock'), timeout=0):
        _run_locked(out, resume)


def _run_locked(out, resume):
    judge_path = ROOT / 'scripts/judge_replies.py'
    spec = importlib.util.spec_from_file_location('local_judge', judge_path)
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)

    baseline_freeze = _read(BASELINE / 'freeze.json')
    result = _read(BASELINE / 'result.json')
    if not result.get('completed') or result['conversations'] != 60 or result['source_ratings'] != 4800:
        raise ValueError('A completed 60-conversation baseline is required')
    baseline_packet = BASELINE / 'packet'
    public, mapping, hashes = judge.load_packet(baseline_packet)
    receipt = _read(baseline_packet / 'ratings/run.json')
    if (receipt['packet_hashes'] != hashes or len(public) != 480 or len(receipt['personas']) != 10
            or receipt['calibration'] or receipt['response_schema'] != judge.SCHEMA
            or receipt['personas'] != baseline_freeze['personas']
            or any(receipt[k] != value for k, value in baseline_freeze['judge'].items())
            or {row['system'] for row in mapping.values()} != {'BM25', 'MiniLM', 'V4', 'V5'}
            or receipt['script_sha256'] != baseline_freeze['files_sha256']['scripts/judge_replies.py']):
        raise ValueError('Completed baseline packet or judge receipt differs from the frozen run')
    for name in ('data/evaluation/personas.csv', 'data/evaluation/judge_rubric.md'):
        if _digest(ROOT / name) != baseline_freeze['files_sha256'][name]:
            raise ValueError('Persona roster or rubric differs from the completed baseline')
    ratings = _load_ratings(baseline_packet / 'ratings', set(mapping), receipt, judge)
    personas = [p['id'] for p in receipt['personas']]
    v6 = _read(BASELINE / 'v6_selections.json')
    exports = read_jsonl(BASELINE / 'outputs.jsonl')
    outputs = {row['item_id']: row['output'] for row in exports}
    if (len(exports) != 60 or len(outputs) != 60 or len(v6) != 120
            or {(s['item_id'], s['style']) for s in v6} != {(i, style) for i in outputs for style in ('STANDARD', 'LOCAL')}):
        raise ValueError('Incomplete frozen V6 selections or exports')
    public_by_id = {row['candidate_id']: row for row in public}
    for selection in v6:
        item, style, cid = selection['item_id'], selection['style'], selection['candidate_id']
        original = outputs[item][style.lower()]['candidate_text']
        if (mapping[cid]['item_id'] != item or public_by_id[cid]['style'] != style
                or public_by_id[cid]['reply'].split() != original.split()
                or selection['system'] != 'V6'):
            raise ValueError('V6 selection does not match the corresponding frozen candidate')

    models = [entry for entry in baseline_freeze['models'] if entry['path'].startswith('artifacts/reranker_qwen/')]
    if not models or any(_digest(ROOT / entry['path']) != entry['sha256'] for entry in models):
        raise ValueError('Frozen Qwen files changed')
    baseline_files = ['freeze.json', 'result.json', 'outputs.jsonl', 'v6_selections.json',
                      'packet/ratings/run.json', 'packet/ratings/ratings.jsonl']
    source_files = [Path(__file__), ROOT / 'scripts/run_final_test.py', judge_path, LEXICON_PATH]
    source_files.extend(sorted((ROOT / 'src/replycoach').rglob('*.py')))
    freeze = dict(record_type='held_out_v7_extension', no_test_tuning=True, seed=SEED,
                  baseline=str(BASELINE.relative_to(ROOT)), packet_hashes=hashes,
                  baseline_files={name: _digest(BASELINE / name) for name in baseline_files},
                  files={path.relative_to(ROOT).as_posix(): _digest(path) for path in source_files},
                  models=models, judge=baseline_freeze['judge'])
    freeze_path = out / 'freeze.json'
    if resume:
        if _read(freeze_path) != freeze:
            raise ValueError('V7 run inputs changed since the freeze')
    else:
        _save(freeze_path, freeze)

    config = judge.judge_settings(judge.read_env(ROOT / '.env'))
    if judge.prepare_judge(config) != freeze['judge']:
        raise ValueError('Gemma model or settings differ from the completed baseline')
    selections_path = out / 'refinements.json'
    refinements = _read(selections_path) if selections_path.exists() else {}
    if not set(refinements) <= set(outputs):
        raise ValueError('Saved refinements contain unexpected conversations')
    if len(refinements) < len(outputs):
        import gc
        import torch
        from replycoach import demo_resources

        torch.set_num_threads(2)
        scorer = demo_resources.get_shared_ranker('qwen_dailydialog', device='cpu')
        for item, output in sorted(outputs.items()):
            if item in refinements:
                continue
            original = Candidate.from_dict(output['local'])
            if original.meta['ranker'] != scorer.identity:
                raise ValueError('Loaded Qwen identity differs from the frozen V6 selection')
            refinement = refine_local_reply(original, scorer, output['memory_speakers'],
                                            rng=random.Random(f'{SEED}:{item}'))
            if not refinement.complete:
                raise RuntimeError(f'V7 refinement failed for {item}: {refinement.error}')
            refinements[item] = refinement.to_dict()
            _save(selections_path, refinements)
            print(f'V7 selections: {len(refinements)}/60', flush=True)
        del scorer
        demo_resources.get_shared_ranker.clear()
        gc.collect()
    winners = {}
    for item, refinement in refinements.items():
        pool = [outputs[item]['local']['candidate_text'], *refinement['variations']]
        selected = refinement['selected']['candidate_text']
        if (not refinement['complete'] or refinement['original'] != outputs[item]['local']
                or len(set(pool)) != 4 or len(pool) != 4 or selected not in pool
                or [r['candidate_text'] for r in refinement['ranked_candidates']][0] != selected):
            raise ValueError('Invalid saved V7 refinement')
        winners[item] = selected
    new_public, new_private, v7 = make_v7_packet(public, mapping, v6, winners)
    selection_freeze = out / 'selection_freeze.json'
    selection_digest = dict(refinements_sha256=_digest(selections_path), selections=v7)
    if selection_freeze.exists():
        if _read(selection_freeze) != selection_digest:
            raise ValueError('Candidate sets or winners changed after selection was frozen')
    else:
        _save(selection_freeze, selection_digest)
    _save(out / 'v7_selections.json', v7)
    packet = out / 'packet'
    if new_public:
        if not packet.exists():
            with TemporaryDirectory(dir=out) as tmp:
                staging = Path(tmp) / 'packet'
                write_judging_packet(staging, new_public, new_private)
                staging.rename(packet)
        pending, private, packet_hashes = judge.load_packet(packet, local_only=True)
        if pending != new_public or any(any(private[r['candidate_id']][k] != val for k, val in r.items())
                                        for r in new_private):
            raise ValueError('Supplemental judging packet differs from saved selections')
        cmd = [sys.executable, str(judge_path), str(packet), '--local-only']
        ratings_folder = packet / 'ratings'
        if (ratings_folder / 'run.json').exists():
            cmd.append('--resume')
        print(f'{len(new_public)} new Local replies × {len(personas)} personas', flush=True)
        subprocess.check_call(cmd, cwd=ROOT)
        manifest = _read(ratings_folder / 'run.json')
        if (manifest['packet_hashes'] != packet_hashes
                or any(manifest[k] != receipt[k] for k in ('personas', 'rubric', 'response_schema', 'calibration', *freeze['judge']))):
            raise ValueError('V7 judgments do not match the baseline evaluator')
        ratings.update(_load_ratings(ratings_folder, set(private), manifest, judge))
    summary, paired, delta = _report(mapping, v6, v7, ratings, personas, [*judge.SCORES, 'would_send'])
    report = dict(completed=True, evaluation_split='test', conversations=60, baseline_ratings=4800,
                  new_local_replies=len(new_public), new_ratings=len(ratings) - 4800,
                  total_judgments=len(ratings), summary=summary, paired_local_v7_minus_v6=delta,
                  paired_local_by_conversation=paired,
                  limitation='Paired descriptive means from correlated AI personas.')
    _save(out / 'result.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'paired_local_by_conversation'}, indent=2))
