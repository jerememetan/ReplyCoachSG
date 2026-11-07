import pytest
from replycoach.generate.sealion_data import source,encode_reply,pad_example

class ToyTokenizer:
    eos_token_id=1
    def encode(self,text,add_special_tokens=False):return [ord(c)+2 for c in text]


def row():return dict(item_id='one',context=['You have my pen.','Yes.'],speakers=['user','peer'],message='Can I return it?',response='Sure.')


def test_conversation_source_has_no_reference_or_intent():
    r=row();r.update(response='SECRET',reply_acts=['SECRET'],expectation='SECRET')
    assert 'SECRET' not in source(r) and source(r).endswith('You (LOCAL):')
    with pytest.raises(ValueError):source({**r,'speakers':[]})


def test_causal_mask_retains_real_eos_but_masks_prompt_and_eos_padding():
    e=encode_reply(row(),ToyTokenizer());n=e['prompt_tokens'];real=len(e['input_ids'])
    padded=pad_example(e,1,real+3)
    assert padded['labels'][:n]==[-100]*n
    assert padded['labels'][real-1]==1
    assert padded['labels'][real:]==[-100]*3
    assert padded['attention_mask'][real:]==[0]*3
    assert any(label!=-100 for label in padded['labels'])


def test_overbudget_reply_is_not_silently_cut():
    with pytest.raises(ValueError,match='truncated'):encode_reply(row(),ToyTokenizer(),limit=3)
