import importlib
import importlib.util
import json
from dataclasses import replace

import pytest

from replycoach.schema import Candidate


def api():
    assert importlib.util.find_spec('replycoach.shared_ranker'), 'Shared ranking feature has not been implemented'
    return importlib.import_module('replycoach.shared_ranker')


def candidate(text, style='STANDARD', source='V5', score=.5):
    return Candidate([], 'incoming', text, style, source, score, meta={'candidate_source': source})


class FakeScorer:
    identity = {'model': 'test-only', 'revision': 'fixed', 'max_length': 512}

    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def score_pairs(self, pairs):
        self.calls.append(pairs)
        return [api().PairScore(self.values.get(reply, -2.), False) for _, reply in pairs]


def test_shared_selection_ignores_source_scores_and_selects_styles_separately():
    m = api()
    proposals = {
        'BM25': [candidate('wrong answer', source='V0', score=1), candidate('can meet lah', 'LOCAL', 'V0', .01)],
        'V5': [candidate('correct answer', score=.001), candidate('wrong local', 'LOCAL', score=.99)],
    }
    scorer = FakeScorer({'correct answer': -1, 'can meet lah': 2})
    result = m.rank_proposals('Can you meet?', [], proposals, scorer)
    assert result.standard.candidate_text == 'correct answer'
    assert result.standard.score == -1  # The legacy .15 fallback threshold must not apply.
    assert result.local.candidate_text == 'can meet lah'
    changed = {source: [replace(c, score=1-c.score) for c in cs] for source, cs in proposals.items()}
    again = m.rank_proposals('Can you meet?', [], changed, scorer)
    assert again.standard.candidate_text == result.standard.candidate_text
    assert again.local.candidate_text == result.local.candidate_text
    assert 'BM25' not in scorer.calls[0][0][0] and 'V5' not in scorer.calls[0][0][0]


def test_duplicate_proposals_are_scored_once_per_style_with_all_origins():
    m = api()
    proposals = {'BM25': [candidate('same reply', source='V0')],
                 'MiniLM': [candidate('same  reply', source='V2')],
                 'V5': [candidate('same reply', 'LOCAL')]}
    result = m.rank_proposals('incoming', [], proposals, FakeScorer())
    assert len(result.candidates) == 2
    assert result.standard.meta['contributing_models'] == ['BM25', 'MiniLM']
    assert len(result.standard.meta['original_proposals']) == 2
    assert result.local.meta['contributing_models'] == ['V5']


def test_ties_and_source_reordering_do_not_prefer_a_model():
    m = api()
    proposals = {'V5': [candidate('Zulu')], 'BM25': [candidate('Alpha', source='V0')]}
    first = m.rank_proposals('incoming', [], proposals, FakeScorer())
    second = m.rank_proposals('incoming', [], dict(reversed(list(proposals.items()))), FakeScorer())
    assert first.standard.candidate_text == second.standard.candidate_text == 'Alpha'
    assert first.local is None


def test_ranking_does_not_mutate_source_candidates_and_keeps_fallback_identity():
    m = api()
    fallback = candidate('Please confirm the time', source='fallback', score=0)
    original = fallback.to_dict()
    result = m.rank_proposals('incoming', [], {'V4': [fallback]}, FakeScorer())
    assert fallback.to_dict() == original
    assert result.standard.source == 'V6'
    assert result.standard.meta['fallback_models'] == ['V4']
    assert result.standard.meta['original_proposals'][0]['origin'] == 'fallback'
    assert result.standard.meta['ranker']['model'] == 'test-only'


def test_query_preserves_consecutive_speakers_and_does_not_guess_unknowns():
    m = api()
    query = m.build_query('third', ['first', 'second'], ['peer', 'peer'])
    assert 'Peer: first\nPeer: second' in query
    assert 'Incoming peer message: third' in query
    assert query.endswith('User reply:')
    unknown = m.build_query('third', ['first', 'second'])
    assert 'Unknown: first\nUnknown: second' in unknown
    with pytest.raises(ValueError, match='speaker'):
        m.build_query('third', ['first'], ['user', 'peer'])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True, '0.5'])
def test_bad_shared_scores_are_rejected(value):
    m = api()
    scorer = FakeScorer()
    scorer.score_pairs = lambda pairs: [m.PairScore(value, False)]
    with pytest.raises(ValueError, match='finite'):
        m.rank_proposals('incoming', [], {'V5': [candidate('answer')]}, scorer)


@pytest.mark.parametrize('count', [0, 2])
def test_missing_or_excess_scores_are_rejected(count):
    m = api()
    scorer = FakeScorer()
    scorer.score_pairs = lambda pairs: [m.PairScore(1, False)] * count
    with pytest.raises(ValueError, match='count'):
        m.rank_proposals('incoming', [], {'V5': [candidate('answer')]}, scorer)


def test_empty_pool_does_not_load_or_call_a_scorer():
    m = api()
    scorer = FakeScorer()
    result = m.rank_proposals('incoming', [], {}, scorer)
    assert not result.candidates and result.standard is None and result.local is None
    assert not scorer.calls
    with pytest.raises(ValueError, match='incoming'):
        m.rank_proposals(' ', [], {}, scorer)


def test_style_aware_selector_scores_each_pool_with_its_requested_style():
    m = api()
    scorer = FakeScorer()
    scorer.identity = {**scorer.identity, 'query_format': 'speaker-conversation-style-v2', 'fine_tuned': True}
    output = m.rank_proposals('incoming', ['first'],
                             {'V5': [candidate('standard'), candidate('local', 'LOCAL')]},
                             scorer, ['peer'])
    pairs = scorer.calls[0]
    assert 'Requested reply style: STANDARD' in pairs[0][0]
    assert 'Requested reply style: LOCAL' in pairs[1][0]
    assert 'Peer: first' in pairs[0][0]
    assert output.standard.meta['selector'] == 'finetuned_cross_encoder'


def test_qwen_score_kind_and_checkpoint_identity_survive_shared_selection():
    scorer = FakeScorer()
    scorer.identity = {'model': 'Qwen/Qwen3-Reranker-0.6B', 'fine_tuned': True,
                       'query_format': 'speaker-conversation-style-v2',
                       'score_kind': 'yes_minus_no_logit', 'adapter_sha256': 'a'*64}
    output = api().rank_proposals('incoming', ['first', 'second'],
                                 {'V5': [candidate('answer')]}, scorer, ['peer', 'peer'])
    assert output.standard.source == 'V6'
    assert output.standard.meta['score_kind'] == 'yes_minus_no_logit'
    assert output.standard.meta['ranker']['adapter_sha256'] == 'a'*64
    assert 'Peer: first\nPeer: second' in scorer.calls[0][0][0]
    assert 'Requested reply style: STANDARD' in scorer.calls[0][0][0]


@pytest.mark.parametrize('mutation', ['smoke', 'no_updates', 'same_weights', 'wrong_format'])
def test_unfinished_or_smoke_checkpoints_are_rejected_before_loading_models(tmp_path, mutation):
    m = api()
    manifest = {'record_type': 'v6_cross_encoder_training', 'completed': True,
                'production_checkpoint': True, 'device': 'cuda', 'weights_changed': True,
                'optimizer_steps': 144, 'base_model': m.MODEL_NAME, 'base_revision': m.MODEL_REVISION,
                'query_format': 'speaker-conversation-style-v2',
                'initial_weights_sha256': 'a'*64, 'best_weights_sha256': 'b'*64}
    if mutation == 'smoke': manifest['production_checkpoint'] = False
    elif mutation == 'no_updates': manifest['optimizer_steps'] = 0
    elif mutation == 'same_weights': manifest['best_weights_sha256'] = 'a'*64
    else: manifest['query_format'] = 'unknown'
    (tmp_path/'v6_training.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='completed GPU|provenance'):
        m.CrossEncoderScorer(checkpoint=tmp_path)
