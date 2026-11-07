from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from prepare_structured_fit_v1 import pick,rotate
from replycoach.generate.structured_rl_v1 import make_cases,reward,prompt
from replycoach.generate.structured_fit_v1 import fit,MODES
from test_train_v5 import tiny_model


def test_frozen_selection_balances_labels_and_rotations_preserve_semantics():
    cases=make_cases();train=pick([r for r in cases if r['split']=='train'])
    assert len(train)==12
    for case in train:
        good=next(a for a in case['actions'] if reward(case,a)==1)
        positions=[]
        for shift in range(4):
            row=rotate(case,shift)
            assert row['facts']==case['facts']
            assert sum(reward(row,a)==1 for a in row['actions'])==1
            positions.append(row['actions'].index(good))
        assert sorted(positions)==[0,1,2,3]


@pytest.mark.parametrize('mode',MODES)
def test_actual_fit_step_and_both_evaluation_paths(tmp_path,mode):
    torch.set_num_threads(1)
    cases=make_cases();train=[c for c in cases if c['split']=='train'][:2]
    novel=[c for c in cases if c['split']=='validation'][:2]
    permuted=[rotate(c,n) for c in train for n in (1,2,3)]
    base=tiny_model(tmp_path/'base',[prompt(c) for c in train+novel+permuted]+list('ABCD'))
    r=fit(base,train,permuted,novel,tmp_path/'run',mode,device='cpu',steps=1,record_steps=(0,1))
    assert r['status']=='complete' and r['optimizer_steps']==1
    assert [e['optimizer_steps'] for e in r['history']]==[0,1]
    assert r['history'][-1]['permuted']['cases']==6
    assert 'gold_token_loss' in r['history'][-1]['train']
    assert all('generated' in o and 'choice_correct' in o for o in r['history'][-1]['novel']['outputs'])
    assert (tmp_path/'run/final/structured_policy.json').exists()
