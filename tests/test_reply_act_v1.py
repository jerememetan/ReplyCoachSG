import pytest
import torch
from transformers import MT5Config, MT5ForConditionalGeneration
from replycoach.generate.reply_act_v1 import ReplyActMT5, REPLY_ACTS, act_vector


def tiny():
    torch.manual_seed(425)
    base = MT5ForConditionalGeneration(MT5Config(vocab_size=32, d_model=16, d_kv=4, d_ff=32,
        num_layers=1, num_decoder_layers=1, num_heads=4, dropout_rate=0,
        pad_token_id=0, eos_token_id=1, decoder_start_token_id=0))
    return ReplyActMT5(base).eval()


def batch():
    return torch.tensor([[3,4,1,0],[5,1,0,0]]), torch.tensor([[1,1,1,0],[1,1,0,0]])


def test_classifier_ignores_padding_and_predicted_route_forbids_reference_acts():
    m = tiny(); ids, mask = batch()
    _, _, logits = m.memory(ids, mask)
    _, _, cropped = m.memory(ids[1:2,:2], mask[1:2,:2])
    torch.testing.assert_close(logits[1], cropped[0], atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match='Reference acts'):
        m.memory(ids, mask, oracle_acts=torch.ones(2,len(REPLY_ACTS)))


def test_predicted_memory_and_joint_losses_receive_gradients():
    m = tiny(); ids, mask = batch()
    enc, extended, _ = m.memory(ids, mask)
    assert enc.last_hidden_state.shape == (2,5,16)
    assert extended.tolist() == [[1,1,1,1,0],[1,1,1,0,0]]
    output = m(ids, mask, torch.tensor([[6,1],[7,1]]), torch.tensor([act_vector(['accept']),act_vector(['decline'])]))
    output['loss'].backward()
    for param in [m.classifier.weight, m.act_embeddings, m.model.shared.weight]:
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum()>0


def test_none_route_equals_base_and_generation_works_for_all_routes():
    m = tiny(); ids, mask = batch(); labels=torch.tensor([[6,1],[7,1]])
    acts=torch.tensor([act_vector(['accept']),act_vector(['decline'])])
    control=m(ids,mask,labels,acts,condition=False,auxiliary=False)
    base=m.model(input_ids=ids,attention_mask=mask,labels=labels).loss
    torch.testing.assert_close(control['loss'],base)
    for route in ['none','predicted','oracle']:
        tokens,prob=m.generate(ids,mask,route,acts if route=='oracle' else None,max_new_tokens=3,num_beams=2)
        assert tokens.shape[0]==2 and prob.shape==(2,len(REPLY_ACTS))


def test_oracle_requires_valid_labels_and_multiact_is_preserved():
    m=tiny(); ids,mask=batch()
    with pytest.raises(ValueError):m.memory(ids,mask,'oracle',torch.zeros(2,len(REPLY_ACTS)))
    assert sum(act_vector(['decline','counterpropose']))==2
    with pytest.raises(ValueError):act_vector(['incoming_invitation'])


def test_reference_act_targets_do_not_change_predicted_decoder_token_loss():
    m=tiny(); ids,mask=batch(); labels=torch.tensor([[6,1],[7,1]])
    accept=torch.tensor([act_vector(['accept'])]*2)
    decline=torch.tensor([act_vector(['decline'])]*2)
    first=m(ids,mask,labels,accept)
    second=m(ids,mask,labels,decline)
    torch.testing.assert_close(first['token_loss'],second['token_loss'])
    assert first['act_loss']!=second['act_loss']


def test_pilot_labels_cover_original_development_splits_and_do_not_enter_source():
    import runpy
    from pathlib import Path
    runner=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_mt5_reply_act_v1.py'))
    train=runner['rows']('train');val=runner['rows']('validation')
    assert len(train)==190 and len(val)==50
    assert not {r['item_id'] for r in train}&{r['item_id'] for r in val}
    for row in train+val:
        act_vector(row['reply_acts'])
        changed={**row,'response':'SECRET_REFERENCE','reply_acts':['SECRET_LABEL']}
        assert runner['source'](changed)==runner['source'](row)
        assert 'SECRET' not in runner['source'](changed)
