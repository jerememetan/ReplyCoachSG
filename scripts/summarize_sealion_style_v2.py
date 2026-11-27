"""Archive the completed inference follow-up and its same-context comparison."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import sealion_style_v1 as style
import sealion_style_guard_v2 as revised


def read(name):
    return json.loads((ROOT / name).read_text())


def save(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    assert Path.home().name == 'hm.teoh.2024'
    follow = read('outputs/combined_copy100_v2/results.json')
    assert follow['status'] == 'complete' and len(follow['outputs']) == 32
    assert follow['source_adapters_unchanged']
    assert follow['raw_checkpoint_replay_differences'] == []
    baseline = {row['item_id']: row for row in read('outputs/algorithms/results.json')['outputs']}
    previous = {row['item_id']: row for row in read('review/all_methods_and_checkpoints.json')}
    rows = []
    for row in follow['outputs']:
        ident = row['item_id']
        assert row['standard_anchor'] == baseline[ident]['standard_anchor']
        texts = dict(previous[ident]['outputs'])
        assert texts['copy_loss_100'] == row['methods']['beam4']['text']
        for method, entry in row['methods'].items():
            texts['copy100_v2_' + method] = entry['text']
        rows.append(dict(item_id=ident, scope=row['scope'], english=row['standard_anchor'],
            context=list(zip(row['speakers'], row['context'], strict=True)),
            message=row['message'], outputs=texts))
    save('review/final_combined_comparison.json', rows)
    fields = ['item_id', 'scope', 'english', 'context', 'message'] + list(rows[0]['outputs'])
    with (ROOT / 'review/final_combined_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(item_id=row['item_id'], scope=row['scope'], english=row['english'],
                context=json.dumps(row['context'], ensure_ascii=False), message=row['message'], **row['outputs']))
    metrics = {}
    for method in rows[0]['outputs']:
        metrics[method] = {}
        for scope in ['reused_development', 'assistant_authored_diagnostic']:
            subset = [row for row in rows if row['scope'] == scope]
            metrics[method][scope] = dict(cases=len(subset),
                exact_copies=sum(style.canonical(row['english']) == style.canonical(row['outputs'][method]) for row in subset),
                heuristic_guard_passes=sum(revised.guard(row['english'], row['outputs'][method])['passed'] for row in subset))
    ownership = next(row for row in rows if row['item_id'] == 'CTXDEV-ownership-2')
    checks = {name: revised.guard(ownership['english'], ownership['outputs'][name])
        for name in ['beam4', 'copy_loss_100', 'copy100_v2_combined']}
    summary = read('summary.json')
    summary['jobs']['combined_copy100_v2'] = 358022
    summary['combined_followup'] = dict(status='complete', checkpoint=follow['checkpoint'],
        checkpoint_sha256=follow['checkpoint_sha256'], cases=32,
        training_performed=False, default_changed=False, original_validation_selected_step=0,
        raw_checkpoint_replay_differences=[], source_adapters_unchanged=True,
        guard='contractions_v2', copy_weight=.5, style_weight=.2,
        grammar_candidates=True, metrics=metrics,
        ownership_example=dict(item_id=ownership['item_id'], english=ownership['english'],
            original=ownership['outputs']['beam4'], raw_copy100=ownership['outputs']['copy_loss_100'],
            combined=ownership['outputs']['copy100_v2_combined'], heuristic_checks=checks),
        limitations='32 reused review cases, not locked test or human accuracy; guard and style metrics are incomplete surface heuristics.')
    summary['review_csv'] = 'review/final_combined_comparison.csv'
    summary['final_analysis_script_sha256'] = sha(Path(__file__))
    save('summary.json', summary)
    archive = ROOT / 'results_only_v2.tgz'
    include = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(ROOT)
        if '__pycache__' in rel.parts or any(part == 'selected' or part.startswith('step_') for part in rel.parts[:-1]):
            continue
        if path.name.startswith('results_archive'):
            continue
        if path.suffix in {'.json', '.csv', '.py', '.out', '.sbatch'}:
            include.append(path)
    with tarfile.open(archive, 'w:gz') as tar:
        for path in include:
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    record = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive),
        files=len(include), weights_included=False)
    save('results_archive_v2.json', record)
    print(json.dumps(dict(archive=record, ownership=summary['combined_followup']['ownership_example'],
        metrics={name: metrics[name] for name in ['beam4', 'copy_loss_100', 'copy100_v2_combined']})), flush=True)


if __name__ == '__main__':
    main()
