import importlib
import pytest
from test_all_models import resources,Generator


def run_api():
    return importlib.import_module('replycoach.sealion_comparison').run_sealion_comparison


def options():
    models,scorer,kwargs=resources()
    kwargs.pop('scorer_factory')
    sea=Generator('SEALION',score=0)
    kwargs['v5_factory']=lambda:sea
    return models,sea,kwargs


def test_direct_replies_preserve_sealion_text_even_with_zero_input_score():
    models,sea,kwargs=options()
    result=run_api()('incoming',['earlier'],speakers=['user'],**kwargs)
    assert result.complete and len(result.proposals)==4
    assert result.standard.candidate_text=='SEALION standard'
    assert result.local.candidate_text=='SEALION local'
    assert len(sea.calls)==1 and all(len(m.calls)==1 for m in models[:3])
    assert result.standard.meta['generation_source']=='SEALION'
    payload=result.to_dict()
    assert payload['selection_method']=='sealion_direct'
    assert payload['reranker'] is None and payload['v7'] is None
    assert payload['standard']['candidate_text']==result.standard.candidate_text


def test_missing_sealion_has_no_silent_substitution():
    _,_,kwargs=options()
    def missing():raise RuntimeError('SEA-LION checkpoint unavailable')
    kwargs['v5_factory']=missing
    result=run_api()('incoming',[],**kwargs)
    assert not result.complete and len(result.proposals)==3
    assert result.standard is None and result.local is None
    assert result.source_errors['V5']=='SEA-LION checkpoint unavailable'


def test_missing_comparison_source_keeps_direct_sealion_replies():
    _,_,kwargs=options()
    def missing():raise RuntimeError('comparison checkpoint unavailable')
    kwargs['v4_factory']=missing
    result=run_api()('incoming',[],**kwargs)
    assert not result.complete and result.standard.source=='SEALION'
    assert result.local.candidate_text=='SEALION local'


def test_invalid_speakers_fail_before_any_model_load():
    models,sea,kwargs=options()
    with pytest.raises(ValueError,match='speaker'):
        run_api()('incoming',['earlier'],speakers=[],**kwargs)
    assert not sea.calls and all(not m.calls for m in models)
