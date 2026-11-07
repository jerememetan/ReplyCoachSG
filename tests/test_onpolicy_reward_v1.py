from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from analyze_onpolicy_reward_v1 import summarize

def fixtures():
    return [dict(id=f'{g}-{n}',item_id=str(g),category='time',draft=f'draft{n}',expected_verdict=verdict,rating=dict(verdict=verdict,reward=reward)) for g in range(3) for n,(verdict,reward) in enumerate([('acceptable',1),('unacceptable',-1)])]

def test_audited_correct_scores_and_contrasts_pass_small_fixed_batch():
    summary=summarize(fixtures(),expected_count=6)
    assert summary['scoped_pilot_gate_pass'] and summary['mixed_reward_groups']==3

def test_any_wrong_positive_fails_even_with_large_otherwise_correct_batch():
    rows=fixtures()*20
    rows[1]={**rows[1],'rating':dict(verdict='acceptable',reward=1)}
    summary=summarize(rows,expected_count=120)
    assert summary['balanced_agreement']>.9 and not summary['scoped_pilot_gate_pass']

def test_all_abstentions_and_incomplete_batch_fail():
    rows=fixtures()
    assert not summarize(rows,expected_count=7)['scoped_pilot_gate_pass']
    for row in rows:row['rating']=dict(verdict='uncertain',reward=None)
    assert not summarize(rows,expected_count=6)['scoped_pilot_gate_pass']
