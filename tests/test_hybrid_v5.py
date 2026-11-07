from pathlib import Path
from types import SimpleNamespace
import pytest
from replycoach.generate.hybrid_v5 import HybridV5Generator
from replycoach.generate.rewrite_v5 import rewrite_standard
from replycoach.generate.style_control import REWRITE_STYLE


def components():
    def part(name,source):return SimpleNamespace(model=name,tokenizer=name+'-tok',device='cpu',max_source=source,max_target=48,format={'speaker_labels':True},checkpoint=Path('/'+name))
    g=object.__new__(HybridV5Generator);g.standard=part('standard-model',192);g.local=part('rewrite-model',256);g.local.format['rewrite_local']=True
    return g


def test_second_model_receives_exact_generated_standard_and_explicit_roles(monkeypatch):
    from replycoach.generate import train_v5
    calls=[]
    def gen(model,tok,rows,style,*args):
        calls.append((model,style,rows,args));return ['I will send it.'] if style=='STANDARD' else ['I send you later.']
    monkeypatch.setattr(train_v5,'generate_replies',gen)
    row=dict(item_id='one',context=['Can you help?'],speakers=['peer'],message='Send it?',response='Reference must not be used.')
    pair=components().generate_rows([row])[0]
    assert calls[0][0]=='standard-model' and calls[1][0]=='rewrite-model'
    assert calls[1][1]==REWRITE_STYLE
    assert calls[1][2][0]['standard_reply']=='I will send it.'
    assert calls[1][2][0]['speakers']==['peer']
    assert pair['local_conditioned_on']==pair['standard']=='I will send it.'
    assert 'standard_reply' not in row


def test_candidate_scores_use_respective_model_tokenizer_and_actual_anchor(monkeypatch):
    from replycoach.generate import rerank_v5
    g=components();g.generate_rows=lambda rows:[dict(standard='I will send it.',local='I send you later.')]
    calls=[]
    def logp(model,tok,rows,style,*args):calls.append((model,tok,rows,style,args));return [-1.]
    monkeypatch.setattr(rerank_v5,'reply_logprobs',logp)
    result=g.generate('Send it?',[])
    assert [c.style for c in result]==['STANDARD','LOCAL']
    assert calls[0][0]=='standard-model' and calls[1][0]=='rewrite-model'
    assert calls[1][2][0]['standard_reply']=='I will send it.' and calls[1][3]==REWRITE_STYLE
    assert result[1].meta['two_checkpoint_pipeline']
    assert result[1].meta['local_conditioned_on']==result[0].candidate_text


def test_empty_or_mismatched_anchors_are_rejected_before_rewrite():
    with pytest.raises(ValueError):rewrite_standard(None,None,[{}],[],192,48,'cpu')
    with pytest.raises(ValueError):rewrite_standard(None,None,[{}],[' '],192,48,'cpu')


def test_loader_honors_explicit_hybrid_configuration(monkeypatch,tmp_path):
    from replycoach.generate import providers,v5_provider,hybrid_v5
    standard=tmp_path/'standard';local=tmp_path/'local'
    for p in (standard,local):p.mkdir();(p/'v5_input.json').write_text('{}')
    monkeypatch.setenv('V5_CHECKPOINT',str(standard));monkeypatch.setenv('V5_LOCAL_REWRITE_CHECKPOINT',str(local))
    seen=[];sentinel=object()
    monkeypatch.setattr(hybrid_v5,'HybridV5Generator',lambda a,b:seen.append((a,b)) or sentinel)
    assert providers.load_v5() is sentinel and seen==[(standard,local)]
    monkeypatch.delenv('V5_LOCAL_REWRITE_CHECKPOINT')
    monkeypatch.setattr(v5_provider,'V5Generator',lambda a:sentinel)
    assert providers.load_v5() is sentinel
