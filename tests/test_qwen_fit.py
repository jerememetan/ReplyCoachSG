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


@pytest.mark.parametrize('epoch_losses,expected_epochs,expected_best',[
    ([.5,.4,.45,.5],4,2),
    ([.5,.4,.45,.39,.4,.41],6,4),
])
def test_twelve_epoch_budget_stops_after_two_nonimproving_checks_and_keeps_best(
        inputs,tmp_path,monkeypatch,epoch_losses,expected_epochs,expected_best):
    m=importlib.import_module('replycoach.qwen_fit')
    bundle=bundle_path(inputs,tmp_path)
    real_loss=torch.nn.functional.binary_cross_entropy_with_logits
    losses=iter(loss for value in epoch_losses for loss in [value,value])
    def controlled_loss(*args,**kwargs):
        if not torch.is_grad_enabled():
            return torch.tensor(next(losses))
        return real_loss(*args,**kwargs)
    monkeypatch.setattr(torch.nn.functional,'binary_cross_entropy_with_logits',controlled_loss)
    result=m.run_training(bundle,tmp_path/'early',source='test-only',smoke=True,
                          epochs=12,early_stopping_patience=2,model_factory=tiny)
    assert result['epochs']==12 and result['epochs_completed']==expected_epochs
    assert result['best_epoch']==expected_best
    assert result['best_validation_loss']==pytest.approx(epoch_losses[expected_best-1])
    assert result['stopped_early'] and result['stop_reason']=='validation_patience'
    assert result['early_stopping_patience']==2
    saved=json.loads((tmp_path/'early/best/qwen_training.json').read_text())
    assert saved['best_epoch']==expected_best and saved['adapter_sha256']==result['adapter_sha256']


def test_twelve_epoch_settings_are_supported_without_changing_smoke_limits(inputs,tmp_path):
    m=importlib.import_module('replycoach.qwen_fit')
    result=m.run_training(bundle_path(inputs,tmp_path),tmp_path/'cap',source='test-only',smoke=True,
                          epochs=12,early_stopping_patience=2,max_steps=1,model_factory=tiny)
    assert result['epochs']==12 and result['epochs_completed']==1
    assert result['stop_reason']=='smoke_step_limit' and not result['production_checkpoint']


def test_second_stage_loads_selected_adapter_and_records_its_hash(inputs,tmp_path):
    m=importlib.import_module('replycoach.qwen_fit')
    bundle=bundle_path(inputs,tmp_path)
    first=m.run_training(bundle,tmp_path/'first',source='test-only',smoke=True,
                         epochs=1,max_steps=1,model_factory=tiny)
    second=m.run_training(bundle,tmp_path/'second',source='test-only',smoke=True,
                          epochs=1,max_steps=1,model_factory=tiny,
                          initial_adapter=tmp_path/'first/best')
    assert second['initialization']['adapter_sha256']==first['adapter_sha256']
    assert second['initial_adapter_parameters_sha256']==first['best_adapter_parameters_sha256']
    assert second['weights_changed'] and second['base_unchanged']
    (tmp_path/'first/best/adapter_model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='adapter'):
        m.run_training(bundle,tmp_path/'third',source='test-only',smoke=True,
                       epochs=1,max_steps=1,model_factory=tiny,
                       initial_adapter=tmp_path/'first/best')


def test_real_half_precision_overflow_skips_update_then_recovers_with_lower_scale():
    m=importlib.import_module('replycoach.qwen_fit')
    assert hasattr(m,'finish_optimizer_step'), 'AMP overflow recovery is not implemented'
    p=torch.nn.Parameter(torch.tensor([1.0]))
    optimizer=torch.optim.AdamW([p],lr=.1,weight_decay=.1)
    scaler=torch.amp.GradScaler('cpu',init_scale=65536.)
    scaler.scale(p.half().sum()).backward()  # finite loss, overflowing FP16 backward
    assert not torch.isfinite(p.grad).all()
    result=m.finish_optimizer_step(optimizer,scaler,[p])
    assert result['updated'] is False
    assert result['loss_scale_after']==32768.
    assert p.item()==1. and not optimizer.state
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(p.half().sum()).backward()
    result=m.finish_optimizer_step(optimizer,scaler,[p])
    assert result['updated'] and p.item()<1.
    assert optimizer.state[p]['step'].item()==1


def test_nonfinite_gradients_without_scaler_still_fail_before_weight_update():
    m=importlib.import_module('replycoach.qwen_fit')
    assert hasattr(m,'finish_optimizer_step'), 'AMP overflow recovery is not implemented'
    p=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.SGD([p],lr=.1)
    p.grad=torch.tensor([float('nan')])
    with pytest.raises(RuntimeError,match='nonfinite.*gradients'):
        m.finish_optimizer_step(optimizer,torch.amp.GradScaler('cpu',enabled=False),[p])
    assert p.item()==1.


def test_overflow_retries_same_window_and_counts_only_successful_updates(inputs,tmp_path,monkeypatch):
    m=importlib.import_module('replycoach.qwen_fit')
    scaler_factory=torch.amp.GradScaler
    monkeypatch.setattr(torch.amp,'GradScaler',lambda *args,**kwargs:scaler_factory('cpu',init_scale=16.))
    real_score=m.score_batch
    calls=[]
    def overflow_once(model,tokenizer,pairs):
        calls.append(pairs)
        if len(calls)==1:
            parameter=next(p for p in model.parameters() if p.requires_grad)
            handle=None
            def inject(grad):
                handle.remove()
                return torch.full_like(grad,float('inf'))
            handle=parameter.register_hook(inject)
        return real_score(model,tokenizer,pairs)
    monkeypatch.setattr(m,'score_batch',overflow_once)
    result=m.run_training(bundle_path(inputs,tmp_path),tmp_path/'retry',source='test-only',
                          smoke=True,epochs=1,max_steps=1,model_factory=tiny)
    assert result['optimizer_steps']==1 and result['gradient_overflow_retries']==1
    assert result['weights_changed'] and result['base_unchanged']
    assert calls[:8]==calls[8:16]  # failed and retried accumulation windows use identical pairs
    events=[json.loads(line) for line in (tmp_path/'retry/overflow_events.jsonl').read_text().splitlines()]
    assert len(events)==1 and events[0]['loss_scale_after']==8.


def test_persistent_overflow_is_bounded_and_never_claims_completion(inputs,tmp_path,monkeypatch):
    m=importlib.import_module('replycoach.qwen_fit')
    assert hasattr(m,'MAX_OVERFLOW_RETRIES'), 'Bounded AMP retries are not implemented'
    scaler_factory=torch.amp.GradScaler
    monkeypatch.setattr(torch.amp,'GradScaler',lambda *args,**kwargs:scaler_factory('cpu',init_scale=16.))
    real_score=m.score_batch
    def always_overflow(model,tokenizer,pairs):
        parameter=next(p for p in model.parameters() if p.requires_grad)
        if not getattr(parameter,'_test_injected',False):
            parameter.register_hook(lambda grad:torch.full_like(grad,float('inf')))
            parameter._test_injected=True
        return real_score(model,tokenizer,pairs)
    monkeypatch.setattr(m,'score_batch',always_overflow)
    monkeypatch.setattr(m,'MAX_OVERFLOW_RETRIES',1)
    with pytest.raises(RuntimeError,match='overflow.*retries'):
        m.run_training(bundle_path(inputs,tmp_path),tmp_path/'bad',source='test-only',
                       smoke=True,epochs=1,max_steps=1,model_factory=tiny)
    assert not (tmp_path/'bad/best/qwen_training.json').exists()
