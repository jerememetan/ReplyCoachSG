import importlib
import importlib.util

import pytest

from replycoach.schema import Candidate
from replycoach.shared_ranker import PairScore


def api():
    assert importlib.util.find_spec('replycoach.all_models'), 'All-model collection has not been implemented'
    return importlib.import_module('replycoach.all_models')


class Retriever:
    def __init__(self, source, prefix):
        self.source, self.prefix, self.calls = source, prefix, []

    def retrieve(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return [Candidate([], message, self.prefix+' standard', 'STANDARD', self.source, .8),
                Candidate([], message, self.prefix+' local', 'LOCAL', self.source, .8)]


class Generator:
    def __init__(self, source, score=.8):
        self.source, self.score, self.calls = source, score, []

    def generate(self, message, context):
        self.calls.append((message, context))
        return [Candidate([], message, self.source+' standard', 'STANDARD', self.source, self.score),
                Candidate([], message, self.source+' local', 'LOCAL', self.source, self.score)]


class Scorer:
    identity = {'model': 'test-only', 'revision': 'fixed'}

    def __init__(self):
        self.calls = []

    def score_pairs(self, pairs):
        self.calls.append(pairs)
        return [PairScore(2 if reply in ['V5 standard', 'BM25 local'] else -2) for _, reply in pairs]


def resources():
    models = [Retriever('V0','BM25'), Retriever('V2','MiniLM'), Generator('V4'), Generator('V5')]
    scorer = Scorer()
    return models, scorer, dict(zip(['bm25_factory','minilm_factory','v4_factory','v5_factory'],
                                    [lambda model=model: model for model in models]), scorer_factory=lambda: scorer)


def test_all_sources_run_once_and_styles_can_choose_different_models():
    m = api()
    models, scorer, kwargs = resources()
    result = m.run_all_models('incoming', ['first','second'], **kwargs)
    assert result.complete and result.available_sources == ['BM25','MiniLM','V4','V5']
    assert all(len(model.calls)==1 for model in models)
    assert len(scorer.calls)==1 and len(scorer.calls[0])==8
    assert result.ranking.standard.meta['contributing_models']==['V5']
    assert result.ranking.local.meta['contributing_models']==['BM25']
    assert not result.source_errors and result.ranking_error is None
    payload = result.to_dict()
    assert payload['record_type']=='all_models_comparison' and payload['complete']
    assert len(payload['proposals'])==4 and len(payload['ranked_candidates'])==8


def test_missing_source_is_visible_and_remaining_sources_are_still_ranked():
    m = api()
    models, scorer, kwargs = resources()

    def missing():
        raise RuntimeError('checkpoint unavailable')

    kwargs['v5_factory']=missing
    result=m.run_all_models('incoming', [], **kwargs)
    assert not result.complete and result.available_sources==['BM25','MiniLM','V4']
    assert result.source_errors=={'V5':'checkpoint unavailable'}
    assert result.ranking.standard is not None
    assert 'V5' not in result.proposals


def test_ranker_failure_retains_all_proposals_without_heuristic_substitution():
    m = api()
    _, _, kwargs=resources()

    def missing_ranker():
        raise RuntimeError('ranker weights unavailable')

    kwargs['scorer_factory']=missing_ranker
    result=m.run_all_models('incoming', [], **kwargs)
    assert len(result.proposals)==4 and result.ranking is None
    assert result.ranking_error=='ranker weights unavailable' and not result.complete
    assert not result.source_errors


def test_source_fallbacks_remain_labelled_and_are_not_counted_as_rnn_generations():
    m=api()
    _, _, kwargs=resources()
    kwargs['v4_factory']=lambda: Generator('V4',score=0)
    result=m.run_all_models('qzxxy', [], memory_mode='M0', **kwargs)
    assert all(c.source=='fallback' for c in result.proposals['V4'])
    assert any('V4' in c.meta['fallback_models'] for c in result.ranking.candidates)


def test_memory_speaker_alignment_preserves_repeated_text_and_real_senders():
    m=api()
    _, scorer, kwargs=resources()
    m.run_all_models('incoming', ['first','same','same'], memory_mode='M1',
                     speakers=['peer','user','peer'], **kwargs)
    query=scorer.calls[0][0][0]
    assert 'Peer: first\nYou: same\nPeer: same' in query


def test_empty_input_and_invalid_speakers_do_not_load_models():
    m=api()
    models, scorer, kwargs=resources()
    with pytest.raises(ValueError,match='incoming'):
        m.run_all_models(' ', [], **kwargs)
    with pytest.raises(ValueError,match='speaker'):
        m.run_all_models('incoming', ['first'], speakers=['peer','user'], **kwargs)
    assert all(not model.calls for model in models) and not scorer.calls


def test_no_available_sources_does_not_call_ranker_or_fabricate_model_outputs():
    m=api()
    scorer=Scorer()
    result=m.run_all_models('incoming', [], scorer_factory=lambda: scorer)
    assert not result.complete and not result.proposals and result.ranking is None
    assert set(result.source_errors)=={'BM25','MiniLM','V4','V5'}
    assert not scorer.calls
