import copy,json
from pathlib import Path
import pytest
import torch
from replycoach.generate.structured_rl_v1 import make_cases,reward,prompt,minute
from replycoach.generate.structured_rl_train_v1 import reinforce_objective,fit
from test_train_v5 import tiny_model


def test_benchmark_splits_and_answer_positions_have_no_overlap_or_position_bias():
    cases=make_cases();groups={}
    for split,count in [('train',192),('validation',48),('test',48)]:
        rows=[c for c in cases if c['split']==split];assert len(rows)==count
        groups[split]={json.dumps(c['facts'],sort_keys=True) for c in rows}
        for category in ('time','availability','ownership'):
            positions=[next(i for i,a in enumerate(c['actions']) if reward(c,a)==1) for c in rows if c['category']==category]
            assert all(positions.count(i)==len(positions)//4 for i in range(4))
        for c in rows:
            assert sum(reward(c,a)==1 for a in c['actions'])==1
            assert c['id'] not in prompt(c) and 'correct' not in prompt(c)
    assert not groups['train']&groups['validation'] and not groups['train']&groups['test'] and not groups['validation']&groups['test']


def test_exact_verifier_rejects_boundary_violations_and_reversed_roles():
    c=dict(category='time',facts=dict(earliest='10:15',latest='11:00'))
    assert reward(c,dict(action='meet',at='10:15'))==1
    assert reward(c,dict(action='meet',at='11:00'))==1
    for value in ('10:14','11:01','99:99','accept'):
        assert reward(c,dict(action='meet',at=value))==-1
    c=dict(category='ownership',facts=dict(owner='user',holder='peer',item='umbrella'))
    assert reward(c,dict(action='return',actor='peer',recipient='user',item='umbrella'))==1
    assert reward(c,dict(action='return',actor='user',recipient='peer',item='umbrella'))==-1
    assert reward(c,dict(action='return',actor='peer',recipient='user',item='cable'))==-1
    assert reward(c,dict(action='refuse'))==-1


def test_reinforce_uses_sampled_reward_and_pushes_probability_toward_valid_action():
    torch.manual_seed(425)
    logits=torch.zeros(1,4,requires_grad=True);p=logits.softmax(-1)
    loss,_=reinforce_objective(p,p.detach(),torch.tensor([[1.,-1.,-1.,-1.]]),group_size=4096)
    loss.backward()
    assert logits.grad[0,0]<0 and (logits.grad[0,1:]>0).all()
    with pytest.raises(ValueError):reinforce_objective(p,p.detach(),torch.ones(1,4),group_size=1)


@pytest.mark.parametrize('arm',['rl','sft'])
def test_tiny_end_to_end_arm_saves_selected_policy_and_checks_test_separately(tmp_path,arm):
    torch.set_num_threads(1)
    cases=make_cases();parts=[[c for c in cases if c['split']==s][:2] for s in ('train','validation','test')]
    base=tiny_model(tmp_path/'base',[prompt(c) for group in parts for c in group]+list('ABCD'))
    report=fit(base,*parts,tmp_path/'run',arm,device='cpu',epochs=1,batch_size=2)
    assert report['status']=='complete' and report['optimizer_steps']==1
    assert report['test']['cases']==2
    assert (tmp_path/'run/best/structured_policy.json').exists()
    assert all('test' not in entry for entry in report['history'])
