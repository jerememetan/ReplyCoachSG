"""Freeze the user-requested accepted prefix, preserving authoring evidence."""
import collections
import hashlib
import json
from pathlib import Path
import sealion_large_bc_data as author

ROOT = Path(__file__).resolve().parents[1]


def main():
    recipe = json.loads((ROOT / 'recipe.json').read_text())
    snapshot = recipe['data']['accepted_snapshot']
    path = ROOT / snapshot['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == snapshot['sha256']
    rows = author.loadlines(path)
    assert len(rows) == snapshot['conversations'] == 548
    assert dict(collections.Counter(r['split'] for r in rows)) == snapshot['splits']
    assert len({r['slot'] for r in rows}) == len(rows)
    plan = json.loads((ROOT / 'data/authoring_plan.json').read_text())
    assert {r['slot'] for r in rows} == {s['slot'] for s in plan['slots']}
    for row in rows:
        assert len(row['drafts']) == 2 and len(row['reviews']) == 4
        assert {(r['draft'], r['reviewer']) for r in row['reviews']} == {
            (i, m) for i in range(2) for m in [author.WRITER, author.CRITIC]}
        assert all(author.passed(r['response']) for r in row['reviews'])
    assert json.loads((ROOT / 'authoring/assistant_sample_review.json').read_text())['proceed'] is True
    assert not (ROOT / 'frozen_data').exists(), 'Preserve existing frozen data; do not overwrite'
    author.freeze(rows)
    manifest_path = ROOT / 'frozen_data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['accepted_snapshot'] = snapshot
    manifest['scope_amendment'] = 'User requested first 548 accepted cases instead of waiting for 5240. Original split assignments retained.'
    author.save(manifest_path, manifest)
    author.save(ROOT / 'authoring/progress.json', dict(
        status='frozen_user_requested_subset', accepted=548, requested=548, pending=0,
        splits=snapshot['splits'], authoring_stopped=True, accepted_snapshot=snapshot))


if __name__ == '__main__':
    main()
