import pytest
import torch
from replycoach.generate.sealion_dual_v1 import item_rows,reply_loss
from replycoach.generate.sealion_data import encode_reply,source


def item():return dict(id='a',split='dev',status='completed',session_memory=[{'speaker':'user','text':'I have your pen.'}],message={'text':'Return it?'},replies={'standard':'Sure, I will return it.','local':'can, i return'})

class Toy:
    eos_token_id=1
    def encode(self,text,add_special_tokens=False):return [ord(c)+2 for c in text]


def test_both_original_styles_and_correct_training_prefixes():
    rows=item_rows([item()]);assert len(rows)==2
    assert rows[0]['style']=='STANDARD' and rows[1]['style']=='LOCAL'
    for row in rows:
        e=encode_reply(row,Toy())
        prefix=Toy().encode(source(row,row['style']))
        assert e['input_ids'][:e['prompt_tokens']]==prefix
        assert e['labels'][:e['prompt_tokens']]==[-100]*len(prefix)
    assert rows[0]['response']==item()['replies']['standard']
    assert rows[1]['response']==item()['replies']['local']
    with pytest.raises(ValueError):item_rows([item(),item()])
    with pytest.raises(ValueError):item_rows([{**item(),'split':'test'}])


def test_fp32_causal_loss_ignores_source_and_matches_shifted_labels():
    logits=torch.zeros(1,4,8,dtype=torch.bfloat16,requires_grad=True)
    labels=torch.tensor([[-100,-100,5,1]])
    loss=reply_loss(logits,labels)
    assert loss.dtype==torch.float32
    torch.testing.assert_close(loss,torch.tensor(8.).log())
    loss.backward()
    assert logits.grad[:,0].abs().sum()==0 and logits.grad[:,1:].abs().sum()>0
