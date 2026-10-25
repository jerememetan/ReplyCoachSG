import json
from pathlib import Path

import pytest

from replycoach.generate.decode_v5 import copies_input, has_local_term, marker_terms, reply_metrics, summary_rows
from replycoach.lexicon import LocalTerm

LEXICON = [LocalTerm("makan", "eat", "lexicon"), LocalTerm("可以", "can", "zh"), LocalTerm("can", "can", "x", apply=False)]


def output(standard: str, local: str, context=("are you free later",), message: str = "want to eat?") -> dict:
    return {"item_id": "X", "context": list(context), "message": message,
            "reference_standard": "Sure, where?", "generated_standard": standard,
            "reference_local": "can lah, where?", "generated_local": local}


def test_markers_match_whole_words_and_cjk_but_skip_unapplied_terms():
    terms = marker_terms(LEXICON)

    assert has_local_term("ok lah see you", terms)
    assert has_local_term("我们可以走", terms)
    assert has_local_term("go makan now", terms)
    assert not has_local_term("Allah is a name", terms)
    assert not has_local_term("I can come", terms)


def test_copies_input_needs_three_words_in_a_row():
    assert copies_input("yes you free later", ["are you free later"], "hi")
    assert not copies_input("free, later maybe", ["are you free later"], "hi")
    assert copies_input("ok want to eat", [], "want to eat?")


def test_reply_metrics_rates():
    rows = [output("Sure, where?", "ok lah"), output("are you free later", "are you free later")]

    metrics = reply_metrics(rows, LEXICON)

    assert metrics["items"] == 2
    assert metrics["local_term_generated_local"] == 0.5
    assert metrics["local_term_reference_local"] == 1.0
    assert metrics["copies_input_generated_local"] == 0.5
    assert metrics["same_reply_both_styles"] == 0.5


def test_summary_skips_reports_without_development_outputs(tmp_path: Path):
    stage2 = tmp_path / "run.json"
    stage2.write_text(json.dumps({"best_validation_loss": 2.5, "validation_outputs": [output("a", "b lah")]}))
    stage1 = tmp_path / "pretrain.json"
    stage1.write_text(json.dumps({"validation_outputs": [{"context": [], "message": "m", "response": "r"}]}))

    rows = summary_rows([stage2, stage1])

    assert [row["run"] for row in rows] == ["run"]
    assert rows[0]["best_validation_loss"] == 2.5


def test_decode_writes_metrics_for_both_styles(tmp_path: Path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from test_finetune_v5 import ITEMS, item_texts, write_items
    from test_train_v5 import tiny_model

    from replycoach.generate.decode_v5 import decode

    checkpoint = Path(tiny_model(tmp_path / "tiny", item_texts()))
    validation = write_items(tmp_path / "validation.jsonl", ITEMS, prefix="ITEM-V")
    decoding = {"num_beams": 2, "no_repeat_ngram_size": 3, "encoder_no_repeat_ngram_size": 3}

    report = decode(checkpoint, "check", decoding, validation, output_dir=tmp_path / "artifacts")

    assert report["decoding"] == decoding
    assert report["metrics"]["items"] == len(ITEMS)
    assert {"generated_standard", "generated_local"} <= set(report["validation_outputs"][0])
    assert (tmp_path / "artifacts" / "decode_check.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        decode(checkpoint, "check", decoding, validation, output_dir=tmp_path / "artifacts")


def test_echo_finds_the_friends_own_plan_said_back_as_the_reply():
    from replycoach.generate.decode_v5 import echo_start

    assert echo_start("Okay, I'll take the wrong bus.", "sorry, I took the wrong bus") == len("Okay, ")
    assert echo_start("okay, i'll upload it later", "paiseh, i'll upload a bit later") is not None
    assert echo_start("okay, i order first", "paiseh, i ordered first") is not None
    assert echo_start("I missed a lot of things", "can tell me what I miss?") == 0
    # Same words without the friend's "I", or a different action after "I", are fine.
    assert echo_start("Yes, still got space.", "still got space?") is None
    assert echo_start("ok, i wait for you", "I am reaching soon, wait for me") is None
    assert echo_start("Sure, I'll be there around 8", "i'll be at lks around 8") is None
    assert echo_start("Sure, I'll send it to you.", "can you send me the notes?") is None


def test_joint_replies_split_back_into_the_two_styles():
    from replycoach.generate.style_control import joint_reply, split_joint

    assert split_joint(joint_reply("Yes, it's open. ", " ya open lah")) == ("Yes, it's open.", "ya open lah")
    assert split_joint("Sure, 2pm works.|LOCAL:can 2pm") == ("Sure, 2pm works.", "can 2pm")
    assert split_joint("Sure | LOCAL: can | LOCAL: can lah") == ("Sure", "can | LOCAL: can lah")
    assert split_joint("Sure, 2pm works.") is None
