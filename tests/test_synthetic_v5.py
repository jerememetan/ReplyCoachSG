import copy
import json

import pytest

from replycoach.generate.synthetic_v5 import load_synthetic


EXAMPLE = {
    "id": "SYN-pilot-001", "session_memory": [{"speaker": "user", "text": "I can wait here."}],
    "message": "I took the wrong bus, sorry!",
    "replies": {"standard": "No worries, let me know when you arrive.",
                "local": "Never mind lah, let me know when you reach."},
    "provenance": {"generator": "test", "prompt_version": "paired-v1", "batch_id": "pilot"},
    "review": {"reviewer": "test reviewer", "meaning_preserved": True, "speaker_correct": True,
               "natural_local": True, "facts_supported": True},
}


def write(path, items):
    path.write_text("".join(json.dumps(item) + "\n" for item in items))
    return path


def test_paired_rows_and_actual_speakers(tmp_path):
    rows = load_synthetic(write(tmp_path / "pairs.jsonl", [EXAMPLE]), [])
    assert [r["style"] for r in rows] == ["STANDARD", "LOCAL"]
    assert all(r["speakers"] == ["user"] and r["item_id"] == EXAMPLE["id"] for r in rows)


@pytest.mark.parametrize("field,value", [("meaning_preserved", False), ("speaker_correct", "true"),
                                         ("reviewer", "")])
def test_review_must_be_explicit(tmp_path, field, value):
    item = copy.deepcopy(EXAMPLE)
    item["review"][field] = value
    with pytest.raises(ValueError, match="review"):
        load_synthetic(write(tmp_path / "pairs.jsonl", [item]), [])


def test_protected_message_blocked_even_with_changed_context(tmp_path):
    protected = [{"item_id": "V1", "context": [], "message": "i took the WRONG bus sorry"}]
    with pytest.raises(ValueError, match="protected"):
        load_synthetic(write(tmp_path / "pairs.jsonl", [EXAMPLE]), protected)


def test_duplicates_and_missing_style_rejected(tmp_path):
    other = copy.deepcopy(EXAMPLE)
    other["id"] = "SYN-other"
    with pytest.raises(ValueError, match="duplicate"):
        load_synthetic(write(tmp_path / "pairs.jsonl", [EXAMPLE, other]), [])
    del other["replies"]["local"]
    with pytest.raises(ValueError, match="both replies"):
        load_synthetic(write(tmp_path / "pairs.jsonl", [other]), [])


@pytest.mark.parametrize("joint", [False, True])
def test_training_adds_paired_synthetic_without_changing_validation(tmp_path, joint):
    from test_train_v5 import tiny_model
    from test_finetune_v5 import ITEMS, item_texts, write_items
    from replycoach.generate.finetune_v5 import finetune

    base = tiny_model(tmp_path / "tiny", item_texts())
    train = write_items(tmp_path / "train.jsonl", ITEMS[:3])
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[3:], prefix="ITEM-V")
    settings = dict(lora_rank=2, lora_alpha=4, epochs=1, batch_size=4, max_source=32,
                    max_target=12, output_dir=tmp_path / "artifacts", joint=joint)
    plain = finetune(base, train, validation, output_name="plain", **settings)
    augmented = finetune(base, train, validation, output_name="synthetic",
                         synthetic_path=write(tmp_path / "pairs.jsonl", [EXAMPLE]), **settings)
    assert augmented["history"][0] == plain["history"][0]
    added = 1 if joint else 2
    assert augmented["train_pairs"] == plain["train_pairs"] + added
    assert augmented["train_pairs_by_kind"]["synthetic"] == added
    assert augmented["synthetic"]["selected_ids"] == [EXAMPLE["id"]]
    assert len(augmented["synthetic"]["sha256"]) == 64
