import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('focus_review', Path(__file__).resolve().parents[1] / 'scripts/review_standard_focus_v1.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


@pytest.mark.parametrize('seed,arms', [(425, review.ARMS), (2, ('control', 'capacity'))])
def test_review_requires_complete_axes_and_keeps_splits_separate(tmp_path, monkeypatch, seed, arms):
    monkeypatch.chdir(tmp_path)
    for split, path in [('validation', 'data/dev/validation.jsonl'), ('diagnostic', 'data/reward_diagnostics/standard_focus_v1.jsonl')]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        item = dict(id=split, session_memory=[], message={'text': 'Can you help?'})
        p.write_text(json.dumps(item) + '\n')
        for arm in arms:
            suffix = '_diagnostic' if split == 'diagnostic' else ''
            row = dict(item_id=split, generated_standard=arm, generated_local=arm + ' lah')
            report = dict(optimizer_steps=784, **{('outputs' if split == 'diagnostic' else 'final_validation_outputs'): [row]})
            Path(f'{arm}_s{seed}{suffix}.json').write_text(json.dumps(report))
    output = Path('review')
    review.prepare(Path('.'), output, arms, seed)
    packet = json.loads((output / 'packet.json').read_text())
    assert len(packet['questions']) == 2
    assert set(packet['questions'][0]['options']) == set('ABCD'[:len(arms)])
    ratings = dict(packet_sha256=hashlib.sha256((output / 'packet.json').read_bytes()).hexdigest(), ratings=[])
    for q in packet['questions']:
        for option in q['options']:
            ratings['ratings'].append(dict(n=q['n'], option=option, standard='pass', local='pass', same_meaning='pass', singlish='fail', reason='Neutral phrasing.'))
    (output / 'ratings.json').write_text(json.dumps(ratings))
    review.score(output)
    result = json.loads((output / 'summary.json').read_text())
    assert set(result['counts']) == {'validation', 'diagnostic'}
    assert all(c['pairs'] == 1 and c['all_pass'] == 0 for arms in result['counts'].values() for c in arms.values())
    ratings['ratings'].pop()
    (output / 'ratings.json').write_text(json.dumps(ratings))
    with pytest.raises(AssertionError):
        review.score(output)
