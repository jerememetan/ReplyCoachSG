import importlib
import json
import pytest
import torch
from test_reranker_fit import bundle_path
from test_reranker_training import inputs


class Tokenizer:
    unk_token_id = None

    def encode(self, text, **kwargs):
        return [ord(word[0]) % 250 + 3 for word in text.split()]

    def pad(self, rows, **kwargs):
        n=max(map(len,rows['input_ids']))
        return {'input_ids':torch.tensor([[0]*(n-len(row))+row for row in rows['input_ids']]),
                'attention_mask':torch.tensor([[0]*(n-len(row))+[1]*len(row) for row in rows['input_ids']])}

    def convert_tokens_to_ids(self, text):
        return {'yes':1,'no':2}[text]

    def save_pretrained(self, path):
        (path/'tokenizer_config.json').write_text('{}')


def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config=Qwen3Config(vocab_size=256,hidden_size=16,intermediate_size=24,
                      num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
    return Qwen3ForCausalLM(config), Tokenizer()


def test_lora_smoke_updates_only_adapters_and_records_nonproduction_provenance(inputs, tmp_path):
    m=importlib.import_module('replycoach.qwen_fit')
    bundle=bundle_path(inputs,tmp_path)
    result=m.run_training(bundle,tmp_path/'out',source='test-only',smoke=True,
                          max_steps=2,epochs=1,model_factory=tiny)
    assert result['optimizer_steps']==2 and result['weights_changed']
    assert result['base_unchanged'] and result['production_checkpoint'] is False
    assert result['trainable_parameters'] > 0
    assert (tmp_path/'out/best/adapter_model.safetensors').is_file()
    saved=json.loads((tmp_path/'out/best/qwen_training.json').read_text())
    assert saved['adapter_sha256']==result['adapter_sha256']


def test_invalid_bundle_and_cpu_production_stop_before_loading(inputs,tmp_path,monkeypatch):
    m=importlib.import_module('replycoach.qwen_fit')
    bundle=bundle_path(inputs,tmp_path)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError,match='GPU'):
        m.run_training(bundle,tmp_path/'out')
    (bundle/'train_pairs.jsonl').write_text('{}')
    with pytest.raises(ValueError,match='hash'):
        m.run_training(bundle,tmp_path/'out')
