import json

from test_all_models import resources


def runner():
    from replycoach import all_models
    assert hasattr(all_models, 'run_v7_all_models'), 'V7 wrapper is not implemented'
    return all_models.run_v7_all_models


def test_v7_keeps_initial_selection_and_source_calls_and_reuses_one_scorer():
    from replycoach.all_models import run_all_models
    _, _, baseline_kwargs = resources()
    baseline = run_all_models('incoming', ['first', 'second'], **baseline_kwargs)
    models, scorer, kwargs = resources()
    loads = []
    kwargs['scorer_factory'] = lambda: loads.append(True) or scorer
    result = runner()('incoming', ['first', 'second'], **kwargs)
    assert result.complete and len(loads) == 1
    assert all(len(model.calls) == 1 for model in models)
    assert [len(pairs) for pairs in scorer.calls] == [8, 4]
    assert result.ranking == baseline.ranking
    assert result.proposals == baseline.proposals
    assert result.ranking.standard == baseline.ranking.standard
    assert result.local_refinement.original == baseline.ranking.local
    assert result.final_local == result.local_refinement.selected
    assert result.final_local.meta['contributing_models'] == ['BM25']
    assert result.local_refinement.complete
    payload = result.to_dict()
    assert payload['local'] == result.final_local.to_dict()
    assert payload['standard'] == baseline.ranking.standard.to_dict()
    assert payload['ranked_candidates'] == baseline.to_dict()['ranked_candidates']
    assert payload['v7']['candidate_count'] == 4
    json.dumps(payload, allow_nan=False)


def test_partial_sources_still_refine_the_available_local_winner():
    _, scorer, kwargs = resources()
    kwargs['v5_factory'] = lambda: (_ for _ in ()).throw(RuntimeError('V5 missing'))
    result = runner()('incoming', [], **kwargs)
    assert not result.complete and result.source_errors == {'V5': 'V5 missing'}
    assert result.local_refinement.complete and len(scorer.calls[-1]) == 4


def test_initial_ranker_failure_has_no_second_pass_or_fabricated_v7_result():
    _, _, kwargs = resources()
    kwargs['scorer_factory'] = lambda: (_ for _ in ()).throw(RuntimeError('ranker missing'))
    result = runner()('incoming', [], **kwargs)
    assert result.ranking is None and result.local_refinement is None
    assert result.ranking_error == 'ranker missing' and result.final_local is None


def test_v7_failure_retains_initial_ranking_and_reports_incomplete_result():
    from replycoach.shared_ranker import PairScore
    _, scorer, kwargs = resources()
    calls = []

    def fail_second_pass(pairs):
        calls.append(pairs)
        if len(calls) == 2:
            raise RuntimeError('second pass failed')
        return [PairScore(1) for _ in pairs]

    scorer.score_pairs = fail_second_pass
    result = runner()('incoming', [], **kwargs)
    assert not result.complete and result.ranking_error is None
    assert result.local_refinement.error == 'second pass failed'
    assert result.final_local is result.ranking.local
    assert len(result.proposals) == 4
    assert result.to_dict()['v7']['error'] == 'second pass failed'


def test_v7_logging_records_final_output_once(monkeypatch):
    from replycoach import all_models
    _, _, kwargs = resources()
    events = []
    monkeypatch.setattr(all_models, 'log_event', lambda event, payload: events.append((event, payload)))
    result = runner()('incoming', [], log=True, **kwargs)
    assert len(events) == 1
    assert events[0][1]['local'] == result.final_local.to_dict()
    assert events[0][1]['v7']['candidate_count'] == 4


def test_lexicon_failure_keeps_original_without_running_a_smaller_candidate_pool(monkeypatch):
    from replycoach import lexicon_refinement
    _, scorer, kwargs = resources()

    def missing_variations(reply):
        raise ValueError('lexicon cannot produce three distinct local variations')

    monkeypatch.setattr(lexicon_refinement, 'generate_local_variants', missing_variations)
    result = runner()('incoming', [], **kwargs)
    assert not result.complete and len(scorer.calls) == 1
    assert result.final_local is result.ranking.local
    assert result.local_refinement.variations == []
    assert result.to_dict()['v7']['candidate_count'] == 1
    assert 'three distinct' in result.local_refinement.error
