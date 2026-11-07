import json
from pathlib import Path
import pytest
import torch
from transformers import MT5Config, MT5ForConditionalGeneration
from replycoach.generate.act_follow_v1 import (
    ActFollowMT5,conversation_source,conditioned_source,supported_act_patterns,
    choose_predicted_acts,per_row_nll,
)
from replycoach.generate.reply_act_v1 import act_vector

ROOT=Path(__file__).resolve().parents[1]


def tiny(mode):
    torch.manual_seed(425)
    base=MT5ForConditionalGeneration(MT5Config(vocab_size=32,d_model=16,d_kv=4,d_ff=32,
        num_layers=1,num_decoder_layers=1,num_heads=4,dropout_rate=0,pad_token_id=0,
        eos_token_id=1,decoder_start_token_id=0))
    return ActFollowMT5(base,mode).eval()


def test_paired_data_group_splits_and_two_distinct_goals():
    folder=ROOT/'data/training_expansion/mt5_act_follow_v1'
    seen=set()
    for split,n in [('train',24),('validation',8),('evaluation',16)]:
        data=json.loads((folder/f'{split}.json').read_text())
        groups={r['conversation_id'] for r in data}
        assert len(groups)==n and len(data)==2*n and not seen&groups
        seen|=groups
        for group in groups:
            a,b=[r for r in data if r['conversation_id']==group]
            assert conversation_source(a)==conversation_source(b)
            assert a['response']!=b['response'] and a['reply_acts']!=b['reply_acts']
            assert conditioned_source(a,a['reply_acts'])!=conditioned_source(b,b['reply_acts'])


def test_sources_exclude_reference_targets_and_incoming_labels():
    row=dict(context=['I finish at six.','Okay.'],speakers=['user','peer'],message='Meet at five?',response='SECRET',reply_acts=['SECRET'],intent='SECRET')
    assert 'SECRET' not in conversation_source(row)
    assert 'SECRET' not in conditioned_source(row,['decline'])
    with pytest.raises(ValueError):conditioned_source(row,['SECRET'])


def test_predicted_joint_acts_come_only_from_training_patterns():
    rows=[dict(reply_acts=['accept']),dict(reply_acts=['decline','counterpropose'])]
    patterns=supported_act_patterns(rows)
    logits=torch.tensor([[-8. if not x else 8. for x in act_vector(['decline','counterpropose'])]])
    assert choose_predicted_acts(logits,patterns)==[['decline','counterpropose']]
    assert len(patterns)==2


@pytest.mark.parametrize('mode',['direct','gated','text'])
def test_joint_losses_backpropagate_without_ambiguous_pair_classifier_labels(mode):
    m=tiny(mode);ids=torch.tensor([[3,4,1],[5,6,1]]);mask=torch.ones_like(ids)
    labels=torch.tensor([[7,1],[8,1]])
    acts=torch.tensor([act_vector(['accept']),act_vector(['decline'])]);wrong=acts.flip(0)
    kwargs=dict(conditioned_ids=ids,conditioned_mask=mask,wrong_acts=wrong,
                wrong_ids=ids.flip(0),wrong_mask=mask,paired_mask=torch.ones(2))
    out=m(ids,mask,labels,acts,torch.zeros(2),**kwargs)
    assert out['act_loss']==0
    out['loss'].backward()
    assert m.classifier.weight.grad is None or m.classifier.weight.grad.abs().sum()==0
    assert m.model.shared.weight.grad.abs().sum()>0
    assert torch.isfinite(out['rank_loss'])
    if mode=='gated':assert m.act_gate.grad.abs()>0


def test_ranking_nll_ignores_padding():
    logits=torch.zeros(1,3,5);labels=torch.tensor([[2,1,-100]])
    torch.testing.assert_close(per_row_nll(logits,labels),torch.tensor([5.]).log())


def test_automatic_goal_selection_cannot_use_reference_acts(monkeypatch):
    import runpy
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    runner=runpy.run_path(str(ROOT/'scripts/run_mt5_act_follow_v1.py'))
    choose=runner['selected_acts']
    row={'reply_acts':['SECRET_REFERENCE'],'response':'SECRET_REPLY'}
    assert choose(row,'predicted',['decline'])==['decline']
    assert choose({},'predicted',['accept'])==['accept']
    assert choose(row,'disabled',['decline'])==['decline']
    assert choose(row,'supplied',['decline'])==['SECRET_REFERENCE']
