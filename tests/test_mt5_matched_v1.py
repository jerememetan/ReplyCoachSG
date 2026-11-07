"""Guard original mT5 weights, training target isolation and adapter updates."""
import json
from pathlib import Path
import torch
import pytest
from peft import LoraConfig,get_peft_model
from replycoach.generate.mt5_pretrained_compat import OriginalMT5,original_config,verify_original_embeddings
from replycoach.generate.mt5_matched_v1 import source,encode,reply_loss,task_rows
ROOT=Path(__file__).resolve().parents[1]

def tiny():
    return OriginalMT5(original_config(dict(vocab_size=32,d_model=16,d_kv=8,d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=2,tie_word_embeddings=False)))

def test_original_embedding_mapping_survives_strict_checkpoint_roundtrip(tmp_path):
    model=tiny();verify_original_embeddings(model)
    assert not torch.equal(model.shared.weight,model.lm_head.weight)
    model.save_pretrained(tmp_path)
    cfg=original_config(json.loads((tmp_path/'config.json').read_text()))
    restored,info=OriginalMT5.from_pretrained(tmp_path,config=cfg,output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info.get('mismatched_keys')
    verify_original_embeddings(restored)
    for name,value in model.state_dict().items():assert torch.equal(value,restored.state_dict()[name]),name
    model.eval();restored.eval();ids=torch.tensor([[3,4,1]])
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids,labels=ids).logits,restored(input_ids=ids,labels=ids).logits)

def test_lora_only_update_preserves_all_pretrained_weights():
    model=get_peft_model(tiny(),LoraConfig(task_type='SEQ_2_SEQ_LM',r=2,lora_alpha=4,target_modules=['q','k','v','o','wi_0','wi_1','wo']))
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    trainable=[p for n,p in model.named_parameters() if p.requires_grad and 'lora_' in n]
    assert len(trainable)==sum(p.requires_grad for p in model.parameters())
    before=[p.detach().clone() for p in trainable];opt=torch.optim.AdamW(trainable,lr=.01)
    ids=torch.tensor([[3,4,1]]);labels=torch.tensor([[6,1,-100]])
    loss=reply_loss(model(input_ids=ids,labels=labels).logits,labels)
    assert torch.isfinite(loss);loss.backward();opt.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,trainable))
    for n,p in model.named_parameters():
        if n in frozen:assert torch.equal(p,frozen[n]),n
    verify_original_embeddings(model.base_model.model)

def test_reply_loss_ignores_padding_and_encoder_has_no_hidden_targets():
    r=dict(item_id='x',context=['You have my cable.'],speakers=['user'],message='Return Monday?',response='SECRET LOCAL TARGET',standard_anchor='Return my cable Monday.',diagnostic_anchor='HIDDEN GOLD')
    text=source(r,'LOCAL_REWRITE','EXPLICIT ACTUAL DRAFT')
    assert 'You: You have my cable.' in text and 'Friend: Return Monday?' in text
    assert 'EXPLICIT ACTUAL DRAFT' in text and r['response'] not in text and r['standard_anchor'] not in text and 'HIDDEN GOLD' not in text
    with pytest.raises(ValueError):source(r,'LOCAL_REWRITE',' ')
    logits=torch.randn(1,3,8);labels=torch.tensor([[2,1,-100]])
    assert torch.allclose(reply_loss(logits,labels),reply_loss(logits[:,:2],labels[:,:2]))

def test_protected_team_rows_and_corrected_training_targets_are_exact():
    old=(ROOT/'data/dev/train.jsonl').read_bytes();raw=(ROOT/'data/training_expansion/generated600_repair_v1/combined_train.jsonl').read_bytes()
    assert raw.startswith(old)
    records=[json.loads(l) for l in raw.splitlines()];expanded=task_rows(records)
    assert len(records)==790 and len(expanded)==1580
    for i,r in enumerate(records):
        assert expanded[2*i]['response']==r['replies']['standard']
        assert expanded[2*i+1]['response']==r['replies']['local']
