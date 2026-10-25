from pathlib import Path

import pytest

from replycoach.generate.rerank_v5 import copied_share, pick

ITEM = {"item_id": "X", "context": ["are you free later"], "message": "i took the wrong bus",
        "reference_standard": "No worries, update me.", "reference_local": "gg, update me when reach"}


def report(standard: list[dict], local: list[dict]) -> dict:
    return {"items": [{**ITEM, "candidates": {"STANDARD": standard, "LOCAL": local}}]}


def candidate(text: str, fit: float, generic: float = -3.0, beam: bool = False) -> dict:
    return {"text": text, "fit": fit, "generic": generic, "beam": beam}


def test_copied_share_counts_three_word_runs_from_the_input():
    assert copied_share("okay i'll take the wrong bus", [], "i took the wrong bus") == 1 / 4
    assert copied_share("no worries", [], "i took the wrong bus") == 0.0
    assert copied_share("are you free later", ITEM["context"], "hi") == 1.0


def test_zero_weights_pick_what_the_model_likes_most():
    standard = [candidate("okay, i'll take the wrong bus", -0.5, beam=True), candidate("no worries, update me", -1.0)]

    (output,) = pick(report(standard, standard))

    assert output["generated_standard"] == "okay, i'll take the wrong bus"
    assert output["beam_kept"] == {"STANDARD": True, "LOCAL": True}


def test_each_weight_moves_the_choice():
    copying = candidate("ok i took the wrong bus too", -0.5, beam=True)
    fresh = candidate("no worries, update me when you reach", -0.9)
    generic = candidate("ok see you later", -0.6, generic=-0.4)
    local = candidate("gg lah, update me", -1.2)

    assert pick(report([copying, fresh], [copying]), copy=2)[0]["generated_standard"] == fresh["text"]
    assert pick(report([generic, fresh], [generic]), mmi=0.5)[0]["generated_standard"] == fresh["text"]
    chosen = pick(report([fresh, local], [fresh, local]), style=1)[0]
    # The marker counts for LOCAL and against STANDARD.
    assert chosen["generated_local"] == local["text"] and chosen["generated_standard"] == fresh["text"]


def test_candidates_hold_the_beam_reply_and_scores(tmp_path: Path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from test_finetune_v5 import ITEMS, item_texts, write_items
    from test_train_v5 import tiny_model

    from replycoach.generate.rerank_v5 import candidates

    checkpoint = Path(tiny_model(tmp_path / "tiny", item_texts()))
    validation = write_items(tmp_path / "validation.jsonl", ITEMS, prefix="ITEM-V")

    result = candidates(checkpoint, "check", samples=3, validation_path=validation, max_source=32, max_target=8,
                        output_dir=tmp_path / "artifacts")

    assert len(result["items"]) == len(ITEMS)
    for item in result["items"]:
        for style in ("STANDARD", "LOCAL"):
            options = item["candidates"][style]
            assert 1 <= len(options) <= 4 and options[0]["beam"]
            assert all(option["fit"] <= 0 and option["generic"] <= 0 for option in options)
    assert len(pick(result)) == len(ITEMS)
    with pytest.raises(ValueError, match="already exists"):
        candidates(checkpoint, "check", samples=3, validation_path=validation, output_dir=tmp_path / "artifacts")
