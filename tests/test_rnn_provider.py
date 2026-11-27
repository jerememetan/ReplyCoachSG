"""Small synthetic weights exercise contracts, never production/model quality."""

import math

import pytest
import torch

from replycoach.generate import BOS, EOS, SPECIAL, Vocab
from replycoach.generate.providers import GeneratorUnavailable, load_generator
from replycoach.generate.rnn_provider import FORMAT, RNNGenerator, STYLE_TOKENS, task_source
from replycoach.generate.seq2seq import Seq2Seq


@pytest.fixture
def task_checkpoint(tmp_path):
    vocab = Vocab.from_dict({t: i for i, t in enumerate([*SPECIAL, *STYLE_TOKENS.values(), "yes"])})
    model = Seq2Seq(len(vocab), 4, 8, use_attention=True)
    # Intentionally prefer UNK/PAD: decoding must mask these without inflating its score.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.decoder.out.bias[1] = 9.0
        model.decoder.out.bias[0] = 8.0
        model.decoder.out.bias[vocab.token_to_id["yes"]] = 1.0
    path = tmp_path / "test_only.pt"
    torch.save({"format": FORMAT, "task_trained": True, "model_state": model.state_dict(),
                "vocab": vocab.to_dict(), "use_attention": True, "embed_dim": 4, "hidden_size": 8,
                "max_source_tokens": 8, "max_target_tokens": 5, "epoch": 1,
                "provenance": {"train_sha256": "test-only", "validation_sha256": "test-only"}}, path)
    return path


def test_long_input_keeps_style_and_recent_message(task_checkpoint):
    generator = RNNGenerator(task_checkpoint)
    ids = task_source(generator.vocab, "LOCAL", "yes", ["unknown " * 100], 6)
    assert len(ids) == 6
    assert ids[:2] == [generator.vocab.token_to_id[BOS], generator.vocab.token_to_id["<local>"]]
    assert ids[-2:] == [generator.vocab.token_to_id["yes"], generator.vocab.token_to_id[EOS]]


def test_provider_returns_both_styles_with_raw_uncalibrated_likelihood(task_checkpoint, monkeypatch):
    monkeypatch.setenv("REPLYCOACH_V4_CHECKPOINT", str(task_checkpoint))
    generator = load_generator("v4")
    candidates = generator.generate("yes", ["yes"])
    assert {c.style for c in candidates} == {"STANDARD", "LOCAL"}
    for candidate in candidates:
        assert candidate.source == "V4" and candidate.item_id is None
        assert "<" not in candidate.candidate_text
        assert 0 < candidate.score < 0.01  # Suppressing UNK must not renormalize confidence.
        assert math.isfinite(candidate.score)
        assert candidate.meta["terminated_with_eos"] is False
        assert len(candidate.candidate_text.split()) <= 4
        assert len(candidate.meta["checkpoint_sha256"]) == 64


def test_smoke_weights_cannot_be_a_production_provider(tmp_path, monkeypatch):
    path = tmp_path / "smoke.pt"
    torch.save({"use_attention": True}, path)
    monkeypatch.setenv("REPLYCOACH_V4_CHECKPOINT", str(path))
    with pytest.raises(GeneratorUnavailable, match="task-trained"):
        load_generator("v4")


def test_task_vocabulary_expansion_preserves_pretrained_rows(task_checkpoint):
    from replycoach.generate.train_task import expand_pretrained
    from replycoach.io import load_items

    saved = torch.load(task_checkpoint, weights_only=True)
    train = load_items()[:1]  # Existing engineering fixture; never production gold.
    model, vocab = expand_pretrained(saved, train, 425)
    for token, idx in saved["vocab"].items():
        assert vocab.token_to_id[token] == idx
    old_size = len(saved["vocab"])
    for name, previous in saved["model_state"].items():
        current = model.state_dict()[name]
        if current.shape != previous.shape:
            current = current[:old_size]
        torch.testing.assert_close(current, previous, rtol=0, atol=0)
