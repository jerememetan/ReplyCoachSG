import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

from test_finetune_v5 import ITEMS, item_texts, write_items
from test_train_v5 import tiny_model

from replycoach.generate import providers
from replycoach.generate.finetune_v5 import finetune
from replycoach.generate.v5_provider import V5Generator, configured_checkpoint
from replycoach.io import ROOT
from replycoach.systems import run_system

SETTINGS = dict(lora_rank=2, lora_alpha=4, epochs=1, batch_size=4, max_source=32, max_target=12)


def trained(tmp_path: Path, name: str, **options) -> Path:
    base = tiny_model(tmp_path / "tiny", [*item_texts(), "BOTH", "|", "LOCAL:", "REPLY"])
    train = write_items(tmp_path / "train.jsonl", ITEMS[:3])
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[3:], prefix="ITEM-V")
    finetune(base, train, validation, output_name=name, output_dir=tmp_path / "artifacts", **SETTINGS, **options)
    return tmp_path / "artifacts" / name / "best"


@pytest.mark.parametrize("options", [{}, {"speaker_labels": True}, {"joint": True}])
def test_v5_answers_in_both_styles_through_the_app_contract(tmp_path: Path, monkeypatch, options):
    checkpoint = trained(tmp_path, "served", **options)
    monkeypatch.setenv("V5_CHECKPOINT", str(checkpoint))

    generator = providers.load_generator("v5")
    candidates = generator.generate("class starting soon", ["where are you"])

    assert isinstance(generator, V5Generator) and generator.source == "V5"
    assert [candidate.style for candidate in candidates] == ["STANDARD", "LOCAL"]
    for candidate in candidates:
        assert candidate.candidate_text.strip() and 0 < candidate.score <= 1
        assert candidate.meta["checkpoint"] == str(checkpoint)
        assert candidate.meta["joint"] == bool(options.get("joint"))
        json.dumps(candidate.to_dict(), allow_nan=False)
    # The same checks the export and the demo run on every generator.
    result = run_system("class starting soon", ["where are you"], "M2", mode="generator_only",
                        generator_factory=lambda: generator)
    assert result.effective_mode == "generator_only" and not result.errors


def test_missing_checkpoint_is_a_clear_role_d_message(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("V5_CHECKPOINT", str(tmp_path / "nowhere"))
    with pytest.raises(providers.GeneratorUnavailable, match="Role D: no trained V5 checkpoint"):
        providers.load_generator("v5")


def test_relative_checkpoint_paths_start_at_the_repo_root(monkeypatch):
    monkeypatch.setenv("V5_CHECKPOINT", "artifacts/some_run/best")
    assert configured_checkpoint() == ROOT / "artifacts" / "some_run" / "best"
    monkeypatch.delenv("V5_CHECKPOINT")
    assert configured_checkpoint() == ROOT / "artifacts" / "spk1_ul025_s425" / "best"
