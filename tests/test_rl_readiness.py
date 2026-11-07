"""Protect diagnostic label isolation, abstention handling and reward consistency."""
import importlib.util
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_rl_readiness import reward_payload,validate_reward,summarize_reward,call,ModelOutputError

def test_gold_and_rationales_never_enter_reward_payload():
    c=dict(conversation=[],message='Can you come?',style='LOCAL',draft='Can.',acceptable=True,group='secret',rationale='gold',id='secret')
    assert set(reward_payload(c))=={'conversation','message','style','draft'}

@pytest.mark.parametrize('verdict,issues',[('acceptable',['wrong_role']),('unacceptable',['none']),('uncertain',['none'])])
def test_inconsistent_reward_is_rejected(verdict,issues):
    with pytest.raises(ValueError):validate_reward(dict(verdict=verdict,issues=issues,reason='Reason'))

def test_uncertainty_is_not_a_training_reward():
    assert validate_reward(dict(verdict='uncertain',issues=['unclear'],reason='Ambiguous'))['reward'] is None

def make_results():
    return [dict(group=g,acceptable=good,rating={'verdict':'acceptable' if good else 'unacceptable'})
            for g in ('constraint','role','choice','new_detail') for good in (True,False) for _ in range(10)]

def test_abstention_counts_as_failure_and_cannot_pass_sensitive_gate():
    r=make_results()
    assert summarize_reward(r)['diagnostic_gate_pass']
    for c in [x for x in r if x['group']=='role' and not x['acceptable']][:2]:c['rating']['verdict']='uncertain'
    out=summarize_reward(r)
    assert out['counts']['abstentions']==2
    assert not out['diagnostic_gate_pass']

def test_rejecting_valid_new_details_fails_even_with_high_overall_accuracy():
    r=make_results()
    for c in [x for x in r if x['group']=='new_detail' and x['acceptable']][:2]:c['rating']['verdict']='unacceptable'
    assert not summarize_reward(r)['diagnostic_gate_pass']

def test_truncated_model_output_keeps_raw_evidence(monkeypatch):
    import run_rl_readiness
    raw={'done':True,'done_reason':'length','message':{'content':'unfinished'}}
    monkeypatch.setattr(run_rl_readiness,'request_json',lambda *a,**k:raw)
    with pytest.raises(ModelOutputError) as err:
        call({'base':'http://localhost:1'},{'judge_model':'local','options':{}},'prompt',{}, {})
    assert err.value.response==raw
