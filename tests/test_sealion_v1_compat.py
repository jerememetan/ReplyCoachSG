import torch
from peft import LoraConfig,get_peft_model
from replycoach.generate.sealion_v1 import MPTConfig,SEALionCausalLM


def test_custom_learned_positions_tied_embedding_and_checkpointed_lora_backward():
    torch.manual_seed(425)
    config=MPTConfig(d_model=64,n_heads=4,n_layers=1,vocab_size=256,max_seq_len=64,
        attn_config={'attn_impl':'torch','alibi':False},init_device='cpu',tie_word_embeddings=True)
    base=SEALionCausalLM(config)
    assert base.transformer.learned_pos_emb and not base.transformer.alibi
    assert hasattr(base.transformer,'wpe')
    assert base.get_input_embeddings() is base.get_output_embeddings()
    m=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=4,lora_alpha=8,
        target_modules=['Wqkv','out_proj','up_proj','down_proj']))
    m.gradient_checkpointing_enable();m.enable_input_require_grads();m.train()
    ids=torch.tensor([[3,4,5,1]])
    out=m(input_ids=ids,attention_mask=torch.ones_like(ids),labels=torch.tensor([[-100,-100,5,1]]))
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if 'lora_' in n)
    assert all(p.grad is None for n,p in m.named_parameters() if 'lora_' not in n)


def test_actual_checkpoint_loading_restores_absent_legacy_default(tmp_path):
    import json
    from replycoach.generate.sealion_v1 import config_for_checkpoint
    config=MPTConfig(d_model=64,n_heads=4,n_layers=1,vocab_size=256,max_seq_len=64,
        attn_config={'attn_impl':'torch','alibi':False},init_device='cpu',tie_word_embeddings=True)
    original=SEALionCausalLM(config)
    original.save_pretrained(tmp_path)
    path=tmp_path/'config.json';values=json.loads(path.read_text())
    values.pop('tie_word_embeddings',None);path.write_text(json.dumps(values))
    restored=config_for_checkpoint(tmp_path)
    assert restored.tie_word_embeddings is True and restored.init_device=='meta'
    loaded,info=SEALionCausalLM.from_pretrained(tmp_path,config=restored,dtype=torch.float32,
        device_map={'':'cpu'},local_files_only=True,output_loading_info=True)
    assert not info.get('missing_keys') and not info.get('unexpected_keys')
    assert not any(p.is_meta for p in loaded.parameters())
    for key,tensor in original.state_dict().items():
        torch.testing.assert_close(tensor,loaded.state_dict()[key],rtol=0,atol=0)
    from replycoach.generate.sealion_v1 import checkpoint_loading_problems
    assert json.loads(json.dumps(checkpoint_loading_problems(info)))==checkpoint_loading_problems(info)


def test_loading_metadata_supports_transformers_set_collections():
    import json
    from replycoach.generate.sealion_v1 import checkpoint_loading_problems
    clean=checkpoint_loading_problems({'missing_keys':set(),'unexpected_keys':set(),
        'mismatched_keys':[],'error_msgs':[]})
    assert json.loads(json.dumps(clean))==clean and not any(clean.values())


def test_legacy_cache_next_logits_match_full_context_for_lora():
    config=MPTConfig(d_model=64,n_heads=4,n_layers=1,vocab_size=256,max_seq_len=64,
        attn_config={'attn_impl':'torch','alibi':False},init_device='cpu',tie_word_embeddings=True)
    m=get_peft_model(SEALionCausalLM(config),LoraConfig(task_type='CAUSAL_LM',r=4,
        lora_alpha=8,target_modules=['Wqkv','out_proj','up_proj','down_proj'])).eval()
    ids=torch.tensor([[3,4,5]])
    with torch.no_grad():
        first=m(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True)
        cached=m(input_ids=torch.tensor([[7]]),attention_mask=torch.ones(1,4,dtype=torch.long),
                 past_key_values=first.past_key_values,use_cache=True).logits[:,-1]
        whole=m(input_ids=torch.tensor([[3,4,5,7]]),attention_mask=torch.ones(1,4,dtype=torch.long),use_cache=False).logits[:,-1]
    torch.testing.assert_close(cached,whole,atol=1e-5,rtol=1e-5)
