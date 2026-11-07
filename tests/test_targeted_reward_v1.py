from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import targeted_reward_v1 as scorer
from run_rl_readiness import ModelOutputError

def test_only_public_fields_are_sent():
    row=dict(context=['I am busy Monday.'],speakers=['user'],message='Meet Monday?',draft='No.',expected_verdicts=['unacceptable'],proposed_reward=-1,reason='hidden',id='secret')
    p=scorer.payload(row)
    assert set(p)=={'conversation','message','style','draft'}
    assert 'hidden' not in str(p) and 'secret' not in str(p)

def test_malformed_model_output_is_no_reward_not_a_negative(monkeypatch):
    monkeypatch.setattr(scorer,'call',lambda *a: (_ for _ in ()).throw(ModelOutputError('length',{'done_reason':'length'})))
    r=scorer.score(dict(context=[],speakers=[],message='Hi',draft='Hello'),{}, {})
    assert r['rating']['reward'] is None and r['raw_failure']=={'done_reason':'length'}

def test_unsafe_scorer_fails_gate_even_if_it_abstains_everywhere():
    rows=[dict(split=split,category=c,expected_verdicts=[label],rating=dict(verdict='uncertain')) for split in ('heldout','actor') for c in ('time','availability','ownership') for label in ('acceptable','unacceptable')]
    s=scorer.summarize(rows)
    assert not s['necessary_diagnostic_pass'] and not s['ready_for_rl']

def test_perfect_small_diagnostic_still_is_not_rl_readiness():
    rows=[dict(split=split,category=c,expected_verdicts=[label],rating=dict(verdict=label)) for split in ('heldout','actor') for c in ('time','availability','ownership') for label in ('acceptable','unacceptable')]
    s=scorer.summarize(rows)
    assert s['necessary_diagnostic_pass'] and not s['ready_for_rl']
