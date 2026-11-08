import json
from pathlib import Path

import pytest

from replycoach.generate.providers import GeneratorUnavailable, load_generator


def test_missing_handoff_has_no_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv('REPLYCOACH_SEALION_BUNDLE', str(tmp_path))
    with pytest.raises(GeneratorUnavailable, match='bundle'):
        load_generator('sealion_handoff')


def test_pair_mapping_preserves_identity_and_rejects_failed_stages():
    from replycoach.generate.handoff_provider import pair_candidates

    pair = dict(english='See you later.', singlish='See you later lah.',
                english_status='complete', singlish_status='complete', standard_anchor='See you later.')
    candidates = pair_candidates(pair, 'SEALION', Path('bundle'), 'Hi', ['Hello'])
    assert [c.style for c in candidates] == ['STANDARD', 'LOCAL']
    assert [c.candidate_text for c in candidates] == ['See you later.', 'See you later lah.']
    for candidate in candidates:
        candidate.validate()
        assert candidate.source == 'SEALION'
        assert candidate.context == ['Hello']
        assert candidate.meta['standard_anchor'] == 'See you later.'
        assert candidate.meta['score_kind'] == 'unscored_bundle_output'
    with pytest.raises(GeneratorUnavailable, match='length_limit'):
        pair_candidates({**pair, 'singlish_status': 'length_limit'}, 'SEALION', Path('bundle'), 'Hi', [])


def test_worker_reuses_pipeline_and_keeps_bundle_imports_isolated(tmp_path):
    from replycoach.generate.handoff_provider import HandoffGenerator

    (tmp_path / 'manifest.json').write_text(json.dumps({'kind': 'mt5', 'files': {}}))
    (tmp_path / 'infer.py').write_text('''
class ReplyPipeline:
    def __init__(self, root, device):
        self.calls = 0
    def generate_pairs(self, rows):
        self.calls += 1
        print('loader progress must not corrupt protocol')
        assert rows[0]['speakers'] == ['peer', 'user']
        assert rows[0]['message'] == '你好, makan?'
        return [dict(english=str(self.calls), singlish='可以 lah', standard_anchor=str(self.calls),
                     english_status='complete', singlish_status='complete')]
''', encoding='utf-8')
    generator = HandoffGenerator(tmp_path, 'mt5')
    try:
        assert generator.generate('你好, makan?', ['one', 'two'])[0].candidate_text == '1'
        result = generator.generate('你好, makan?', ['one', 'two'])
        assert result[0].candidate_text == '2'
        assert result[1].candidate_text == '可以 lah'
    finally:
        generator.close()
