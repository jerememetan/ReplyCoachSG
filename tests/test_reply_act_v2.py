import pytest
import torch
from transformers import MT5Config, MT5ForConditionalGeneration
from replycoach.generate.reply_act_v2 import GatedReplyActMT5
from replycoach.generate.reply_act_v1 import act_vector


def setup():
    torch.manual_seed(425)
    base=MT5ForConditionalGeneration(MT5Config(vocab_size=32,d_model=16,d_kv=4,d_ff=32,
        num_layers=1,num_decoder_layers=1,num_heads=4,dropout_rate=0,
        pad_token_id=0,eos_token_id=1,decoder_start_token_id=0))
    m=GatedReplyActMT5(base).eval()
    ids=torch.tensor([[3,4,1,0],[5,1,0,0]]); mask=ids.ne(0)
    acts=torch.tensor([act_vector(['accept']),act_vector(['decline','counterpropose'])])
    return m,ids,mask,acts


def test_zero_gate_preserves_base_memory_loss_and_generation_for_both_routes():
    m,ids,mask,acts=setup()
    original=m.model.get_encoder()(input_ids=ids,attention_mask=mask).last_hidden_state
    labels=torch.tensor([[6,1],[7,1]])
    original_loss=m.model(input_ids=ids,attention_mask=mask,labels=labels).loss
    original_tokens=m.model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=3,num_beams=2)
    for route in ['predicted','oracle']:
        encoded,actual_mask,_=m.memory(ids,mask,route,acts if route=='oracle' else None)
        torch.testing.assert_close(encoded.last_hidden_state,original,rtol=0,atol=0)
        assert torch.equal(actual_mask,mask)
        out=m(ids,mask,labels,acts,training_route=route)
        torch.testing.assert_close(out['token_loss'],original_loss,rtol=0,atol=0)
        tokens,_=m.generate(ids,mask,route,acts if route=='oracle' else None,max_new_tokens=3,num_beams=2)
        assert torch.equal(tokens,original_tokens)


def test_gate_can_learn_and_nonzero_gate_keeps_padding_and_distinguishes_acts():
    m,ids,mask,acts=setup();labels=torch.tensor([[6,1],[7,1]])
    m(ids,mask,labels,acts)['loss'].backward()
    assert torch.isfinite(m.act_gate.grad) and m.act_gate.grad.abs()>0
    assert m.classifier.weight.grad.abs().sum()>0
    with torch.no_grad():m.act_gate.fill_(0.1)
    oracle,_,_=m.memory(ids,mask,'oracle',acts)
    swapped,_,_=m.memory(ids,mask,'oracle',acts.flip(0))
    base,_,_=m.memory(ids,mask,'none')
    assert not torch.allclose(oracle.last_hidden_state,swapped.last_hidden_state)
    torch.testing.assert_close(oracle.last_hidden_state[~mask],base.last_hidden_state[~mask],rtol=0,atol=0)
    m.zero_grad();m(ids,mask,labels,acts)['loss'].backward()
    assert m.act_embeddings.grad.abs().sum()>0


def test_oracle_targets_cannot_enter_normal_prediction_even_after_gate_learns():
    m,ids,mask,acts=setup()
    with torch.no_grad():m.act_gate.fill_(0.2)
    labels=torch.tensor([[6,1],[7,1]])
    first=m(ids,mask,labels,acts)
    second=m(ids,mask,labels,acts.flip(0))
    torch.testing.assert_close(first['token_loss'],second['token_loss'],rtol=0,atol=0)
    with pytest.raises(ValueError,match='Reference acts'):
        m.memory(ids,mask,'predicted',acts)


def test_mixed_training_routes_and_blank_rewrite_is_preserved(monkeypatch):
    import runpy
    from pathlib import Path
    from types import SimpleNamespace
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    runner=runpy.run_path(str(root/'scripts/run_mt5_reply_act_v2.py'))
    choose=runner['training_route']
    assert {choose(True,i,1) for i in range(4)}=={'predicted','oracle'}
    assert all(choose(False,i,1)=='predicted' for i in range(4))
    fn=runner['add_rewrites'];calls=[]
    def rewrite(model,tok,rows,anchors,*args):
        assert all(a.strip() for a in anchors)
        calls.extend(anchors)
        return ['okay']*len(anchors)
    monkeypatch.setitem(fn.__globals__,'rewrite_standard',rewrite)
    monkeypatch.setitem(fn.__globals__,'dump',lambda *args:None)
    data=[dict(item_id=str(i),message='Hi',context=[],speakers=[]) for i in range(2)]
    items=[dict(item_id='0',standard='  '),dict(item_id='1',standard='That works.')]
    result={'arms':{'mixed':{'outputs':{'validation':{'oracle':items}}}}}
    local=SimpleNamespace(model=None,tokenizer=lambda text:{'input_ids':[1]},max_source=192,max_target=48)
    fn(result,local,{'validation':data})
    assert calls==['That works.']
    assert items[0]['rewrite_status']=='blocked_empty_standard' and 'local' not in items[0]
    assert items[1]['local_standard_anchor']=='That works.' and items[1]['rewrite_status']=='complete'
