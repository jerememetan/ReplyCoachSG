from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from analyze_onpolicy_reward_v2 import analyze

def inputs():
    old=[];current=[]
    for i in range(48):
        label='acceptable' if i%2==0 else 'unacceptable'
        row=dict(id=f'old-{i}',item_id=f'old-group-{i//4}',context=['Context'],message='Incoming',category='time',draft=f'draft{i}',expected_verdict=label,rating=dict(verdict=label,reward=1 if i%2==0 else -1))
        old.append(row)
        current.append({**row,'audit_partition':'regression'})
        current.append({**row,'id':f'new-{i}','item_id':f'new-group-{i//4}','audit_partition':'fresh'})
    old[0]={**old[0],'rating':dict(verdict='unacceptable',reward=-1)}
    return dict(status='complete',results=current),dict(status='complete',results=old)

def test_fresh_scores_are_not_merged_with_prompt_development_cases():
    current,old=inputs();r=analyze(current,old)
    assert r['fresh']['samples']==48 and r['regression_repaired_ids']==['old-0']
    assert r['fresh']['scoped_pilot_gate_pass']

def test_overlap_and_incomplete_run_are_rejected():
    current,old=inputs();current['status']='running'
    with pytest.raises(ValueError,match='Full batch'):analyze(current,old)
    current['status']='complete';current['results'][1]['item_id']='old-group-0'
    with pytest.raises(ValueError,match='overlap'):analyze(current,old)
