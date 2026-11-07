"""Checks for second-model pair review without leaking author/reviewer metadata."""
import importlib,json
from pathlib import Path
import pytest


@pytest.fixture
def reviewer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('review_synthetic_pairs')


def case():
    return dict(id='SYN-hidden',scenario_group='hidden',session_memory=[{'speaker':'user','text':'I can carry it.'}],
                message='Thanks!',replies={'standard':'You are welcome.','local':'No worries ah.'},
                provenance={'generator':'hidden'},review={'reviewer':'hidden'})


@pytest.mark.parametrize('verdict,issues,approved',[
    ('acceptable',['none'],True),('acceptable',['wrong_role'],False),
    ('reject',['wrong_role'],False),('uncertain',['none'],False),
])
def test_only_consistent_acceptable_verdict_promotes_pair(reviewer,monkeypatch,verdict,issues,approved):
    def request(base,route,payload):
        task=json.loads(payload['messages'][1]['content'])
        assert set(task)=={'conversation','incoming_peer_message','standard_draft','local_draft'}
        assert 'hidden' not in json.dumps(task)
        return {'done':True,'message':{'content':json.dumps({'verdict':verdict,'reason':'Review.','issues':issues})}}
    monkeypatch.setattr(reviewer,'request_json',request)
    result=reviewer.review_pair(case(),{'base':'unused'},{'judge_model':'fake','think':False,'options':{}})
    assert result['approved']==approved


def test_truncated_review_never_approves(reviewer,monkeypatch):
    monkeypatch.setattr(reviewer,'request_json',lambda *args:{'done':True,'done_reason':'length'})
    with pytest.raises(ValueError,match='Incomplete'):
        reviewer.review_pair(case(),{'base':'unused'},{'judge_model':'fake','think':False,'options':{}})


def test_register_review_checks_both_inputs_and_preserves_targets(reviewer,monkeypatch):
    import copy
    original=case(); rewritten=copy.deepcopy(original)
    rewritten['session_memory'][0]['text']='i can carry it'
    rewritten['message']='thx ah'
    def request(base,route,payload):
        task=json.loads(payload['messages'][1]['content'])
        assert task['original_peer_message']=='Thanks!'
        assert task['incoming_peer_message']=='thx ah'
        assert 'may change only' in payload['messages'][0]['content']
        return {'done':True,'message':{'content':json.dumps({'reason':'Same meaning.','verdict':'acceptable','issues':['none']})}}
    monkeypatch.setattr(reviewer,'request_json',request)
    settings={'judge_model':'fake','think':False,'options':{}}
    assert reviewer.review_pair(rewritten,{'base':'unused'},settings,original)['approved']
    rewritten['replies']['standard']='I will do something different.'
    with pytest.raises(ValueError,match='preserve'):
        reviewer.review_pair(rewritten,{'base':'unused'},settings,original)
