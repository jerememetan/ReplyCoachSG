import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

from test_train_v5 import COSEM, DAILYDIALOG, tiny_model, write_pairs
from transformers import AutoTokenizer

from replycoach.generate.finetune_v5 import finetune, item_rows, shorter_contexts
from replycoach.generate.train_v5 import encode_rows

ITEMS = [
    (["hey are you free this weekend", "yea sure, why?"], "our group jio us to eat, you steady?",
     "Sure, count me in!", "steady lah"),
    (["where are you"], "class starting soon", "I am on my way now.", "otw liao"),
    ([], "can lend me your notes?", "Sure, I will send them later.", "can can later send"),
    (["the canteen closed"], "eat where then?", "How about the food court?", "food court lor"),
]


def write_items(path: Path, items, split: str = "dev", prefix: str = "ITEM-T") -> Path:
    rows = [
        {"id": f"{prefix}-{index:03d}", "split": split,
         "session_memory": [{"speaker": "peer", "text": text} for text in memory],
         "message": {"text": message}, "replies": {"standard": standard, "local": local}}
        for index, (memory, message, standard, local) in enumerate(items, start=1)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def item_texts() -> list[str]:
    return [text for memory, message, standard, local in ITEMS for text in [*memory, message, standard, local]]


def test_item_rows_gives_one_row_per_style(tmp_path: Path):
    rows = item_rows(write_items(tmp_path / "train.jsonl", ITEMS[:1]))

    assert [(row["style"], row["response"]) for row in rows] == [("STANDARD", "Sure, count me in!"),
                                                                 ("LOCAL", "steady lah")]
    assert rows[0]["context"] == rows[1]["context"] == ["hey are you free this weekend", "yea sure, why?"]

    tokenizer = AutoTokenizer.from_pretrained(tiny_model(tmp_path / "tiny", item_texts()))
    (standard, _), (local, _) = encode_rows(rows, tokenizer, max_source=64, max_target=16)
    assert tokenizer.decode(standard).startswith("STANDARD context: hey are you free this weekend | yea sure, why?")
    assert tokenizer.decode(local).startswith("LOCAL context:")


def test_item_rows_refuses_non_development_items(tmp_path: Path):
    with pytest.raises(ValueError, match="only development items"):
        item_rows(write_items(tmp_path / "test.jsonl", ITEMS, split="test"))


def test_lora_finetune_trains_only_the_adapter(tmp_path: Path):
    base = tiny_model(tmp_path / "tiny", item_texts())
    train = write_items(tmp_path / "train.jsonl", ITEMS[:3])
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[3:], prefix="ITEM-V")
    settings = dict(lora_rank=2, lora_alpha=4, epochs=2, batch_size=2, max_source=32, max_target=12,
                    output_dir=tmp_path / "artifacts")

    report = finetune(base, train, validation, output_name="tiny_lora", **settings)

    assert report["method"] == "lora"
    assert (report["train_items"], report["train_pairs"], report["validation_items"]) == (3, 6, 1)
    assert 0 < report["trainable_parameters"] < report["parameter_count"] / 5
    assert [entry["epoch"] for entry in report["history"]] == [0, 1, 2]
    assert {"validation_loss_standard", "validation_loss_local"} <= set(report["history"][0])
    best = tmp_path / "artifacts" / "tiny_lora" / "best"
    assert (best / "adapter_config.json").exists() and not (best / "config.json").exists()
    [output] = report["validation_outputs"]
    assert (output["item_id"], output["reference_local"]) == ("ITEM-V-001", "food court lor")
    assert isinstance(output["generated_standard"], str) and isinstance(output["generated_local"], str)
    with pytest.raises(ValueError, match="already exists"):
        finetune(base, train, validation, output_name="tiny_lora", **settings)


def test_rank_zero_is_a_full_finetune(tmp_path: Path):
    base = tiny_model(tmp_path / "tiny", item_texts())
    train = write_items(tmp_path / "train.jsonl", ITEMS[:3])
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[3:], prefix="ITEM-V")

    report = finetune(base, train, validation, lora_rank=0, epochs=1, batch_size=2, learning_rate=1e-4,
                      max_source=32, max_target=12, output_name="tiny_full", output_dir=tmp_path / "artifacts")

    assert report["method"] == "full" and report["lora"] is None
    assert report["trainable_parameters"] == report["parameter_count"]
    assert (tmp_path / "artifacts" / "tiny_full" / "best" / "config.json").exists()


def test_finetune_refuses_items_in_both_sets(tmp_path: Path):
    train = write_items(tmp_path / "train.jsonl", ITEMS)
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[:1])
    with pytest.raises(ValueError, match="both training and validation"):
        finetune("unused", train, validation, output_dir=tmp_path / "artifacts")


def test_shorter_contexts_keep_the_reply_and_drop_old_turns():
    row = {"item_id": "X", "style": "LOCAL", "context": ["a b", "c d"], "message": "m", "response": "r"}

    assert [variant["context"] for variant in shorter_contexts(row)] == [["c d"], []]
    assert all(variant["response"] == "r" and variant["style"] == "LOCAL" for variant in shorter_contexts(row))
    assert [variant["context"] for variant in shorter_contexts({**row, "context": ["c d"]})] == [[]]
    assert shorter_contexts({**row, "context": []}) == []


def test_augmentation_changes_training_but_not_validation(tmp_path: Path):
    base = tiny_model(tmp_path / "tiny", item_texts())
    train = write_items(tmp_path / "train.jsonl", ITEMS[:3])
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[3:], prefix="ITEM-V")
    public = [write_pairs(tmp_path / "dd.jsonl", "dailydialog", DAILYDIALOG),
              write_pairs(tmp_path / "cosem.jsonl", "cosem", COSEM)]
    settings = dict(lora_rank=2, lora_alpha=4, epochs=1, batch_size=4, max_source=32, max_target=12,
                    output_dir=tmp_path / "artifacts")

    plain = finetune(base, train, validation, output_name="plain", **settings)
    augmented = finetune(base, train, validation, output_name="augmented", context_variants=True,
                         rehearsal_paths=public, rehearsal_count=5, **settings)

    # Items have 2, 1 and 0 context turns, so their pairs gain 2, 1 and 0 shorter copies per style.
    assert augmented["train_pairs_by_kind"] == {"items": 6, "context_variants": 6, "rehearsal": 5}
    assert augmented["train_pairs"] == 17 and augmented["train_items"] == 3
    assert plain["train_pairs_by_kind"] == {"items": 6, "context_variants": 0, "rehearsal": 0}
    assert augmented["history"][0] == plain["history"][0]
