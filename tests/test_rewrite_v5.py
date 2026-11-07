import json
from pathlib import Path
import pytest

from replycoach.generate.rewrite_v5 import rewrite_rows,generate_pair
from replycoach.generate.style_control import mt5_input,REWRITE_STYLE

def pairs():
    return [dict(item_id='I',context=['I finish at six.'],speakers=['user'],message='Seven okay?',style=s,response=t)
            for s,t in [('STANDARD',"Seven works. I'll meet you downstairs."),('LOCAL',"Seven can. I meet you downstairs.")]]

def test_training_pairs_have_one_shared_standard_anchor():
    rows=rewrite_rows(pairs())
    assert rows[0]['style']=='STANDARD'
    assert rows[1]['style']==REWRITE_STYLE and rows[1]['standard_reply']==rows[0]['response']
    assert rows[1]['response']==pairs()[1]['response']
    with pytest.raises(ValueError):rewrite_rows(pairs()+pairs())
    with pytest.raises(ValueError):rewrite_rows(pairs()[:1])

def test_rewrite_source_requires_anchor_and_preserves_actual_roles():
    text=mt5_input(REWRITE_STYLE,'Seven okay?',['I finish at six.'],True,['user'],standard_reply='Seven works.')
    assert 'You: I finish at six.' in text and 'Friend: Seven okay?' in text
    assert 'Standard draft: Seven works. | Singlish rewrite:' in text
    with pytest.raises(ValueError):mt5_input(REWRITE_STYLE,'Hi')
    with pytest.raises(ValueError):mt5_input('STANDARD','Hi',standard_reply='Wrong mode')

def test_inference_anchors_to_generated_reply_not_reference(monkeypatch):
    from replycoach.generate import train_v5
    calls=[]
    def generate(model,tokenizer,rows,style,*args):
        calls.append((style,rows))
        return ['Generated standard.'] if style=='STANDARD' else ['Generated local.']
    monkeypatch.setattr(train_v5,'generate_replies',generate)
    out=generate_pair(None,None,[pairs()[0]],128,48,'cpu')
    assert out=={'STANDARD':['Generated standard.'],'LOCAL':['Generated local.']}
    assert calls[1][0]==REWRITE_STYLE
    assert calls[1][1][0]['standard_reply']=='Generated standard.'
    assert calls[1][1][0]['standard_reply']!=pairs()[0]['response']

def test_empty_standard_is_recorded_without_attempting_empty_anchor(monkeypatch):
    from replycoach.generate import train_v5
    calls=[]
    def generate(*args):calls.append(args);return ['']
    monkeypatch.setattr(train_v5,'generate_replies',generate)
    assert generate_pair(None,None,[pairs()[0]],128,48,'cpu')=={'STANDARD':[''],'LOCAL':['']}
    assert len(calls)==1

def test_encoder_refuses_to_cut_off_chosen_standard(tmp_path):
    from test_train_v5 import tiny_model
    from transformers import AutoTokenizer
    from replycoach.generate.train_v5 import encode_rows
    tokenizer=AutoTokenizer.from_pretrained(tiny_model(tmp_path/'tiny'))
    row=rewrite_rows(pairs())[1]
    row['context']=['long '*100]; row['message']='message '*50
    with pytest.raises(ValueError,match='refusing to truncate'):
        encode_rows([row],tokenizer,24,48,speaker_labels=True)

def test_final_checkpoint_and_report_keep_rewrite_mode(tmp_path):
    from test_train_v5 import tiny_model
    from test_finetune_v5 import ITEMS,item_texts,write_items
    from replycoach.generate.finetune_v5 import finetune,input_format
    base=tiny_model(tmp_path/'tiny',[*item_texts(),'LOCAL_REWRITE','Standard draft:','Singlish rewrite:'])
    train=write_items(tmp_path/'train.jsonl',ITEMS[:3])
    validation=write_items(tmp_path/'validation.jsonl',ITEMS[3:],prefix='ITEM-V')
    report=finetune(base,train,validation,lora_rank=2,lora_alpha=4,epochs=1,batch_size=4,max_source=96,max_target=12,
                    output_name='rewrite',output_dir=tmp_path/'artifacts',speaker_labels=True,rewrite_local=True,save_final=True)
    for name in ('best','final'):
        assert input_format(tmp_path/'artifacts/rewrite'/name)['rewrite_local'] is True
    for key in ('validation_outputs','final_validation_outputs'):
        assert all(c['local_conditioned_on']==c['generated_standard'] for c in report[key])
    assert 'oracle' in report['local_validation_conditioning']

def test_provider_scores_local_under_its_actual_generated_anchor(monkeypatch):
    from replycoach.generate.v5_provider import V5Generator
    from replycoach.generate import rewrite_v5,rerank_v5
    g=object.__new__(V5Generator);g.checkpoint=Path('/fake');g.model=None;g.tokenizer=None;g.device='cpu'
    g.max_source=192;g.max_target=48;g.format={'speaker_labels':True,'joint':False,'rewrite_local':True}
    monkeypatch.setattr(rewrite_v5,'generate_pair',lambda *a,**k:{'STANDARD':['Seven works.'],'LOCAL':['Seven can.']})
    seen=[]
    def logp(model,tokenizer,rows,style,*args):seen.append((rows,style));return [-1.0]
    monkeypatch.setattr(rerank_v5,'reply_logprobs',logp)
    c=g.generate('Seven okay?',[])
    assert [x.candidate_text for x in c]==['Seven works.','Seven can.']
    assert seen[1][1]==REWRITE_STYLE and seen[1][0][0]['standard_reply']=='Seven works.'
    assert c[1].meta['local_conditioned_on']=='Seven works.'


def test_standard_emphasis_preserves_rewrite_anchors():
    from replycoach.generate.rewrite_v5 import emphasize_standard
    standard = {'style': 'STANDARD', 'response': 'I will send it.'}
    local = {'style': 'LOCAL_REWRITE', 'standard_reply': 'I will send it.', 'response': 'I send you lah.'}
    rows = emphasize_standard([standard, local], 3)
    assert rows == [standard, standard, standard, local]
    assert emphasize_standard([standard, local], 1) == [standard, local]
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            emphasize_standard([standard, local], invalid)
