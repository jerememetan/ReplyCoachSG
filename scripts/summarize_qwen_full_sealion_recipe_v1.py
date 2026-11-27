"""Preserve full-pipeline evidence and actual-anchor outputs without inventing ratings."""
import csv
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = {}
    for name, relative in [('public', 'outputs/public/training.json'),
                           ('task', 'outputs/qwen/task/training.json'),
                           ('rewrite', 'outputs/qwen/rewrite/training.json')]:
        p = ROOT / relative
        d = json.loads(p.read_text()) if p.exists() else {}
        reports[name] = {k: d.get(k) for k in ['status', 'selected_epoch', 'optimizer_updates', 'total_updates', 'final_adapter_sha256', 'frozen_base_parameters_unchanged']}
    summary = dict(status='needs_attention', stages=reports, quality_scored=False, default_changed=False,
                   scope='Previously reused development cases; no independent human benchmark.')
    if all(d['status'] == 'complete' for d in reports.values()):
        review = ROOT / 'review'
        review.mkdir(exist_ok=True)
        for policy in ['selected', 'epoch_08']:
            data = json.loads((ROOT / 'outputs/qwen' / ('pipeline_' + policy + '.json')).read_text())
            assert len(data['outputs']) == 75
            rows = []
            for r in data['outputs']:
                assert r['english']['reply'] == r['standard_anchor']
                rows.append(dict(item_id=r['item_id'], context=json.dumps(r['context'], ensure_ascii=False),
                                 message=r['message'], english=r['english']['reply'], singlish=r['singlish']['reply'],
                                 english_status=r['english']['status'], singlish_status=r['singlish']['status']))
            with (review / ('pipeline_' + policy + '.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary[policy + '_flags'] = data['flags']
        summary['status'] = 'complete_unscored'
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    archive = ROOT / 'results_only.tgz'
    if archive.exists():
        raise FileExistsError('Preserve prior evidence collection')
    files = {}
    for p in ROOT.rglob('*'):
        relative = p.relative_to(ROOT)
        if not p.is_file() or p.is_symlink() or relative.parts[0] in ['hf_cache', 'tmp']:
            continue
        if p.suffix in ['.safetensors', '.pt', '.pth', '.bin', '.tgz', '.gz']:
            continue
        if p.stat().st_size > 10 * 1024**2:
            continue
        files[str(relative)] = hashlib.sha256(p.read_bytes()).hexdigest()
    with tarfile.open(archive, 'w:gz') as stream:
        for relative in sorted(files):
            stream.add(ROOT / relative, arcname=relative, recursive=False)
    (ROOT / 'collection_manifest.json').write_text(json.dumps(dict(files=files, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()), indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
