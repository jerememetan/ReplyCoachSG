import pytest
pytest.importorskip('transformers');pytest.importorskip('peft')
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from test_train_v5 import tiny_model
from replycoach.generate.fit_probe_v1 import checked_encoding,fit

ROWS=[dict(item_id='F1',style='STANDARD',context=[],speakers=[],message='Can you send the notes?',response='I will send them.'),dict(item_id='F2',style='STANDARD',context=['I cannot leave before three.'],speakers=['user'],message='Meet at two?',response='Could we meet after three?')]

def base_path(tmp_path):
    return tiny_model(tmp_path/'base',[t for r in ROWS for t in [*r['context'],r['message'],r['response']]])

@pytest.mark.parametrize('method',['lora','full'])
def test_fit_updates_only_intended_parameters_and_records_generation(tmp_path,method):
    base=base_path(tmp_path);original=AutoModelForSeq2SeqLM.from_pretrained(base).shared.weight.detach().clone()
    report,model,tokenizer=fit(base,ROWS,tmp_path/method,method,steps=2,batch_size=2,device='cpu',record_steps=(0,2))
    assert report['optimizer_steps']==2 and report['training_style']=='STANDARD'
    assert [x['optimizer_steps'] for x in report['history']]==[0,2]
    assert len(report['train_generations']['2'])==2
    if method=='lora':
        assert report['trainable_parameters']<report['total_parameters']
        assert torch.equal(original,model.get_base_model().shared.weight)
    else:
        assert report['trainable_parameters']==report['total_parameters']
        assert not torch.equal(original,model.shared.weight)
    assert (tmp_path/method/'final/v5_input.json').exists()


def test_nonstandard_or_truncated_probe_inputs_are_rejected(tmp_path):
    tokenizer=AutoTokenizer.from_pretrained(base_path(tmp_path))
    with pytest.raises(ValueError,match='STANDARD-only'):checked_encoding([{**ROWS[0],'style':'LOCAL'}],tokenizer)
    with pytest.raises(ValueError,match='untruncated'):checked_encoding(ROWS,tokenizer,max_target=1)
