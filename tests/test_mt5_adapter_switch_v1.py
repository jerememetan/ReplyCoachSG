"""Exercise real adapter updates, isolation, switching and serialization."""

import torch
from replycoach.generate.mt5_pretrained_compat import OriginalMT5, original_config
from replycoach.generate.mt5_adapter_switch_v1 import (
    activate_inference, load_switching_adapters, new_task_adapter, task_adapter_config,
)


def tiny():
    return OriginalMT5(original_config(dict(
        vocab_size=32, d_model=16, d_kv=8, d_ff=32,
        num_layers=1, num_decoder_layers=1, num_heads=2,
        tie_word_embeddings=False, dropout_rate=0.0,
    )))


def logits(model):
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([[3, 4, 1]])
        return model(input_ids=ids, labels=ids).logits.detach().clone()


def update(model):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.02)
    ids = torch.tensor([[3, 4, 1]])
    loss = model(input_ids=ids, labels=torch.tensor([[6, 7, 1]])).loss
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_task_update_and_switch_leave_base_and_other_adapter_unchanged():
    torch.manual_seed(425)
    model = new_task_adapter(tiny())
    update(model)
    english = logits(model)
    model.add_adapter("singlish", task_adapter_config())
    model.set_adapter("singlish")
    untouched = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    before = logits(model)
    update(model)
    assert not torch.equal(before, logits(model))
    for name, p in model.named_parameters():
        if name in untouched:
            assert torch.equal(p, untouched[name]), name
    activate_inference(model, "default")
    torch.testing.assert_close(logits(model), english, atol=0, rtol=0)
    assert not any(p.requires_grad for p in model.parameters())
    activate_inference(model, "singlish")
    assert model.active_adapters == ["singlish"]


def test_independent_adapters_reload_on_same_base_and_switch_exactly(tmp_path):
    torch.manual_seed(426)
    base = tiny()
    initial = {n: p.detach().clone() for n, p in base.state_dict().items()}
    english = new_task_adapter(base)
    update(english)
    expected_english = logits(english)
    english.save_pretrained(tmp_path / "english")

    other = tiny()
    other.load_state_dict(initial)
    singlish = new_task_adapter(other)
    update(singlish)
    update(singlish)
    expected_singlish = logits(singlish)
    singlish.save_pretrained(tmp_path / "singlish")

    restored = tiny()
    restored.load_state_dict(initial)
    loaded = load_switching_adapters(restored, tmp_path / "english", tmp_path / "singlish")
    torch.testing.assert_close(logits(loaded), expected_english, atol=0, rtol=0)
    activate_inference(loaded, "singlish")
    torch.testing.assert_close(logits(loaded), expected_singlish, atol=0, rtol=0)
    activate_inference(loaded, "english")
    torch.testing.assert_close(logits(loaded), expected_english, atol=0, rtol=0)
    assert not any(p.requires_grad for p in loaded.parameters())
