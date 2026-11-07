import pytest
import torch

from replycoach.generate.mt5_pretrained_compat import OriginalMT5, original_config
from replycoach.generate.mt5_output_heads_v1 import configure_head, load_head, save_head, train_head_mode


def tiny():
    return OriginalMT5(original_config(dict(
        vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=1,
        num_decoder_layers=1, num_heads=2, tie_word_embeddings=False, dropout_rate=0.1,
    )))


def logits(model):
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([[3, 4, 1]])
        return model(input_ids=ids, labels=ids).logits.detach().clone()


@pytest.mark.parametrize("variant", ["linear", "mlp"])
def test_only_output_parameters_update_and_saved_heads_switch_exactly(tmp_path, variant):
    torch.manual_seed(425)
    model = tiny().eval()
    original_logits = logits(model)
    model = configure_head(model, variant, bottleneck=8)
    torch.testing.assert_close(logits(model), original_logits, atol=0, rtol=0)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    params = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params) == (512 if variant == "linear" else 280)
    save_head(model, tmp_path / "before", variant, {})
    train_head_mode(model)
    assert not model.encoder.training and not model.decoder.training
    ids = torch.tensor([[3, 4, 1]])
    opt = torch.optim.AdamW(params, lr=0.02)
    loss = model(input_ids=ids, labels=torch.tensor([[6, 7, 1]])).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    changed = logits(model)
    assert not torch.equal(changed, original_logits)
    for name, p in model.named_parameters():
        if name in frozen:
            assert torch.equal(p, frozen[name]), name
            assert p.grad is None
    save_head(model, tmp_path / "trained", variant, {})
    load_head(model, tmp_path / "before", variant)
    torch.testing.assert_close(logits(model), original_logits, atol=0, rtol=0)
    load_head(model, tmp_path / "trained", variant)
    torch.testing.assert_close(logits(model), changed, atol=0, rtol=0)
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("variant", ["linear", "mlp"])
def test_head_supports_cached_beam_generation(variant):
    model = configure_head(tiny(), variant, bottleneck=8).eval()
    ids = torch.tensor([[3, 4, 1]])
    with torch.no_grad():
        a = model.generate(input_ids=ids, num_beams=4, max_new_tokens=5, use_cache=True)
        b = model.generate(input_ids=ids, num_beams=4, max_new_tokens=5, use_cache=False)
    assert torch.equal(a, b)


@pytest.mark.parametrize("variant", ["linear", "mlp"])
def test_training_runner_selects_reloadable_head_without_copying_backbone(tmp_path, monkeypatch, variant):
    from pathlib import Path
    from replycoach.generate.train_v5 import collate
    from replycoach.generate.mt5_matched_v1 import reply_loss
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_mt5_head_generation_v1 as run

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return {"input_ids": [3, 4, 1] if text.startswith("STANDARD") else [6, 7, 1]}

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "OUT", tmp_path / "artifacts")
    monkeypatch.setattr(run, "R", dict(recipe=dict(variant=variant, trainable_parameters=512 if variant == "linear" else 280,
                                                  learning_rates=[3e-5, 3e-4]),
                                     source_manifest=dict(files_sha256={"model.safetensors": "tiny-fixture"}), phases={}, selected={}))
    monkeypatch.setattr(run.core, "TOK", Tokenizer())
    monkeypatch.setattr(run.core, "tensor", lambda batch: collate(batch, 0))

    @torch.no_grad()
    def validation(model, examples):
        model.eval()
        x, mask, labels = collate(examples, 0)
        loss = reply_loss(model(input_ids=x, attention_mask=mask, labels=labels).logits, labels)
        return dict(token_ce=float(loss), examples=len(examples), reply_tokens=int(labels.ne(-100).sum()))

    monkeypatch.setattr(run.core, "validation", validation)
    monkeypatch.setattr(run, "fresh_model", lambda: configure_head(tiny(), variant, bottleneck=8))
    monkeypatch.setattr(run, "clear", lambda: None)
    examples = [dict(item_id=f"fixture-{i}", message="Hello", context=[], speakers=[], style="STANDARD", response="Reply") for i in range(2)]
    selected = run.select_stage("standard", examples, examples)
    assert selected.is_symlink() and (selected / "head.safetensors").is_file()
    assert len(run.R["phases"]) == 2
    expected = min(p["best_validation"]["token_ce"] for p in run.R["phases"].values())
    assert run.R["selected"]["standard"]["best_validation"]["token_ce"] == expected
    loaded = load_head(configure_head(tiny(), variant, bottleneck=8), selected, variant)
    assert not any(p.requires_grad for p in loaded.parameters())
