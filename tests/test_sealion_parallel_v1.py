import copy,json,sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_sealion_full_v1 import restore_adapter
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
from replycoach.generate.sealion_v1 import MPTConfig,SEALionCausalLM

def test_targeted_splits_do_not_overlap_or_reverse_ownership():
    folder=ROOT/'data/training_expansion/sealion_fidelity_v1';parts={k:json.loads((folder/f'{k}.json').read_text()) for k in ['train','validation','challenge']}
    sig=lambda r:json.dumps([r['context'],r['message'],r['standard_anchor']])
    assert not {sig(r) for r in parts['train']}&{sig(r) for r in parts['validation']+parts['challenge']}
    for records in parts.values():
        for r in records:
            assert r['value'].casefold() in r['response'].casefold()
            if r['kind']=='ownership':assert (('my ' if r['direction']=='return_to_me' else 'your ')+r['object']) in r['response'].casefold()

def test_checkpoint_restores_exact_adapter_and_next_adam_update():
    def model():
        cfg=MPTConfig(d_model=32,n_heads=4,n_layers=1,vocab_size=64,max_seq_len=32,attn_config={'attn_impl':'torch','alibi':False},init_device='cpu',tie_word_embeddings=True)
        return get_peft_model(SEALionCausalLM(cfg),LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,lora_dropout=0,target_modules=['Wqkv','out_proj','up_proj','down_proj']))
    torch.manual_seed(425);a=model();torch.manual_seed(425);b=model()
    pa=[p for p in a.parameters() if p.requires_grad];pb=[p for p in b.parameters() if p.requires_grad]
    oa=torch.optim.AdamW(pa,lr=1e-3);ob=torch.optim.AdamW(pb,lr=1e-3)
    ids=torch.tensor([[3,4,5,1]])
    def update(m,opt):
        m.train();opt.zero_grad(set_to_none=True);loss=m(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,use_cache=False).loss;loss.backward();opt.step();return loss.detach()
    update(a,oa);state={k:v.detach().clone() for k,v in get_peft_model_state_dict(a).items()};restore_adapter(b,state);ob.load_state_dict(copy.deepcopy(oa.state_dict()))
    la=update(a,oa);lb=update(b,ob);torch.testing.assert_close(la,lb,rtol=0,atol=0)
    for k,v in get_peft_model_state_dict(a).items():torch.testing.assert_close(v,get_peft_model_state_dict(b)[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='shape/key'):restore_adapter(b,{})
