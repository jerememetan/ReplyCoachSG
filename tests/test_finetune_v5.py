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
    row = {"item_id": "X", "style": "LOCAL", "context": ["a b", "c d"], "speakers": ["peer", "user"], "message": "m",
           "response": "r"}

    assert [variant["context"] for variant in shorter_contexts(row)] == [["c d"], []]
    assert [variant["speakers"] for variant in shorter_contexts(row)] == [["user"], []]
    assert all(variant["response"] == "r" and variant["style"] == "LOCAL" for variant in shorter_contexts(row))
    assert [variant["context"] for variant in shorter_contexts({**row, "context": ["c d"], "speakers": ["user"]})] == [[]]
    assert shorter_contexts({**row, "context": [], "speakers": []}) == []


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
    assert augmented["train_pairs_by_kind"] == {"items": 6, "context_variants": 6, "context_pairs": 0, "rehearsal": 5}
    assert augmented["train_pairs"] == 17 and augmented["train_items"] == 3
    assert plain["train_pairs_by_kind"] == {"items": 6, "context_variants": 0, "context_pairs": 0, "rehearsal": 0}
    assert augmented["history"][0] == plain["history"][0]


def test_speaker_labels_use_real_speakers_and_guess_only_without_them():
    from replycoach.generate.style_control import mt5_input

    context = ["where are u at?", "i just reached mall", "wya"]
    # The user double-texted: the real speakers keep both of their turns as You.
    assert mt5_input("LOCAL", "still otw", context, True, ["peer", "user", "user"]) == (
        "LOCAL Friend: where are u at? | You: i just reached mall | You: wya | Friend: still otw | You:"
    )
    # Plain strings, as the app passes them: the guess alternates back from the message.
    assert mt5_input("LOCAL", "still otw", context, True) == (
        "LOCAL You: where are u at? | Friend: i just reached mall | You: wya | Friend: still otw | You:"
    )
    assert mt5_input("STANDARD", "hi", [], speaker_labels=True) == "STANDARD Friend: hi | You:"
    assert mt5_input("STANDARD", "hi", ["a"]) == "STANDARD context: a message: hi"
    with pytest.raises(ValueError, match="speakers for"):
        mt5_input("STANDARD", "hi", ["a", "b"], True, ["peer"])


def test_item_rows_keep_each_turns_speaker(tmp_path: Path):
    path = write_items(tmp_path / "items.jsonl", ITEMS[:1])

    assert [row["speakers"] for row in item_rows(path)] == [["peer", "peer"], ["peer", "peer"]]


def test_speaker_labels_are_saved_with_the_checkpoint_and_used_to_decode(tmp_path: Path):
    from replycoach.generate.decode_v5 import decode
    from replycoach.generate.finetune_v5 import input_format

    base = tiny_model(tmp_path / "tiny", [*item_texts(), "Friend:", "You:"])
    train = write_items(tmp_path / "train.jsonl", ITEMS[:3])
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[3:], prefix="ITEM-V")
    settings = dict(lora_rank=2, lora_alpha=4, epochs=1, batch_size=4, max_source=32, max_target=12,
                    output_dir=tmp_path / "artifacts")

    plain = finetune(base, train, validation, output_name="plain", **settings)
    labelled = finetune(base, train, validation, output_name="labelled", speaker_labels=True, seed=7, **settings)

    assert labelled["speaker_labels"] is True and labelled["seed"] == 7
    assert input_format(tmp_path / "artifacts" / "labelled" / "best") == {"speaker_labels": True, "joint": False}
    assert input_format(tmp_path / "artifacts" / "plain" / "best") == {"speaker_labels": False, "joint": False}
    assert input_format(tmp_path / "missing") == {"speaker_labels": False, "joint": False}
    # Same items, different input text, so the untrained model already scores them differently.
    assert labelled["history"][0] != plain["history"][0]
    report = decode(tmp_path / "artifacts" / "labelled" / "best", "labelled", {"num_beams": 1}, validation,
                    max_source=32, max_target=12, output_dir=tmp_path / "artifacts")
    assert report["speaker_labels"] is True


def test_finetune_refuses_to_drop_speaker_labels_the_base_was_trained_with(tmp_path: Path):
    base = Path(tiny_model(tmp_path / "tiny", [*item_texts(), "Friend:", "You:"]))
    (base / "v5_input.json").write_text('{"speaker_labels": true}', encoding="utf-8")
    train = write_items(tmp_path / "train.jsonl", ITEMS[:3])
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[3:], prefix="ITEM-V")
    with pytest.raises(ValueError, match="add --speaker-labels"):
        finetune(str(base), train, validation, output_dir=tmp_path / "artifacts")


def test_dropping_old_turns_drops_their_speakers_too(tmp_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(tiny_model(tmp_path / "tiny", [*item_texts(), "Friend:", "You:"]))
    row = {"style": "LOCAL", "context": ["hey are you free this weekend", "yea sure, why?"],
           "speakers": ["peer", "user"], "message": "class starting soon", "response": "otw liao"}
    full = len(encode_rows([row], tokenizer, 64, 12, speaker_labels=True)[0][0])

    (source, _), = encode_rows([row], tokenizer, full - 1, 12, speaker_labels=True)

    assert tokenizer.decode(source, skip_special_tokens=True).startswith("LOCAL You: yea sure, why?")


def test_context_pairs_take_user_turns_that_answer_a_peer(tmp_path: Path):
    from replycoach.generate.finetune_v5 import context_pairs

    path = tmp_path / "items.jsonl"
    turns = [("peer", "free this weekend?"), ("user", "yea sure, why?"), ("user", "wait"), ("peer", "eat?"),
             ("user", "ok can")]
    path.write_text(json.dumps({
        "id": "ITEM-T-001", "split": "dev", "session_memory": [{"speaker": s, "text": t} for s, t in turns],
        "message": {"text": "where?"}, "replies": {"standard": "Anywhere.", "local": "anywhere lor"},
    }) + "\n", encoding="utf-8")

    rows = context_pairs(path)

    # The double text "wait" answers no one, so two pairs, each seeing only the turns before its message.
    assert [(row["message"], row["response"]) for row in rows] == [("free this weekend?", "yea sure, why?"),
                                                                  ("eat?", "ok can")]
    assert rows[1]["context"] == ["free this weekend?", "yea sure, why?", "wait"]
    assert rows[1]["speakers"] == ["peer", "user", "user"]
    assert {row["style"] for row in rows} == {"REPLY"}


def test_context_pairs_join_training_only(tmp_path: Path):
    base = tiny_model(tmp_path / "tiny", [*item_texts(), "REPLY"])
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({
        "id": "ITEM-T-001", "split": "dev",
        "session_memory": [{"speaker": "peer", "text": "where are you"}, {"speaker": "user", "text": "on the bus"}],
        "message": {"text": "class starting soon"}, "replies": {"standard": "I am on my way now.", "local": "otw liao"},
    }) + "\n", encoding="utf-8")
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[3:], prefix="ITEM-V")

    report = finetune(base, train, validation, lora_rank=2, lora_alpha=4, epochs=1, batch_size=4, max_source=32,
                      max_target=12, output_dir=tmp_path / "artifacts", use_context_pairs=True)

    assert report["train_pairs_by_kind"]["context_pairs"] == 1 and report["train_pairs"] == 3
    assert report["validation_items"] == 1


def write_candidates(path: Path, candidates: dict[str, dict[str, list[str]]]) -> Path:
    """A rerank_v5 candidates file: {item id: {style: [reply, ...]}}."""
    items = [{"item_id": item_id, "candidates": {style: [{"text": text} for text in texts]
                                                 for style, texts in styles.items()}}
             for item_id, styles in candidates.items()]
    path.write_text(json.dumps({"items": items}), encoding="utf-8")
    return path


ECHO_ITEMS = [
    ([], "sorry, I took the wrong bus", "No worries, tell me when you reach.", "no worries lah"),
    ([], "i ordered first because i was hungry", "That's fine, i'll order when i arrive", "ok i order later"),
]


def test_echo_negatives_keep_echoes_and_skip_rows_where_the_rule_flags_the_team_reply(tmp_path: Path):
    from replycoach.generate.finetune_v5 import echo_negatives

    train = write_items(tmp_path / "train.jsonl", ECHO_ITEMS)
    candidates = write_candidates(tmp_path / "candidates.json", {
        "ITEM-T-001": {"STANDARD": ["Okay, I'll take the wrong bus.", "No worries, see you soon."],
                       "LOCAL": ["ok i take wrong bus also"]},
        "ITEM-T-002": {"STANDARD": ["okay, i'll order first"], "LOCAL": ["okay, i order first"]},
    })

    negatives, skipped = echo_negatives(candidates, item_rows(train))

    # Item 2's team replies also say "i ... order", so the rule cannot tell good from bad there.
    assert skipped == 2
    assert [(row["style"], row["response"]) for row in negatives] == [
        ("STANDARD", "Okay, I'll take the wrong bus."), ("LOCAL", "ok i take wrong bus also")]
    assert negatives[0]["penalise_from"] == len("Okay, ")
    assert negatives[0]["message"] == "sorry, I took the wrong bus"


def test_echo_negatives_refuse_items_outside_the_training_file(tmp_path: Path):
    from replycoach.generate.finetune_v5 import echo_negatives

    train = write_items(tmp_path / "train.jsonl", ECHO_ITEMS)
    candidates = write_candidates(tmp_path / "candidates.json", {"ITEM-V-001": {"STANDARD": [], "LOCAL": []}})
    with pytest.raises(ValueError, match="outside the training file"):
        echo_negatives(candidates, item_rows(train))


def test_negatives_penalise_only_from_the_echo(tmp_path: Path):
    from replycoach.generate.finetune_v5 import encode_negatives

    tokenizer = AutoTokenizer.from_pretrained(tiny_model(tmp_path / "tiny", ["Okay, I'll take the wrong bus."]))
    row = {"style": "STANDARD", "context": [], "message": "sorry", "response": "Okay, I'll take the wrong bus.",
           "penalise_from": len("Okay, ")}

    [(_, target, skip)] = encode_negatives([row], tokenizer, max_source=16, max_target=12)

    assert tokenizer.decode(target[skip:-1]) == "I'll take the wrong bus."
    assert encode_negatives([row], tokenizer, max_source=16, max_target=2) == []  # the echo was cut off


def test_unlikelihood_training_reports_its_negatives_and_penalty(tmp_path: Path):
    texts = [text for memory, message, standard, local in ECHO_ITEMS for text in [message, standard, local]]
    base = tiny_model(tmp_path / "tiny", [*item_texts(), *texts, "Okay,", "I'll", "take", "wrong", "bus."])
    train = write_items(tmp_path / "train.jsonl", ECHO_ITEMS)
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[3:], prefix="ITEM-V")
    candidates = write_candidates(tmp_path / "candidates.json", {
        "ITEM-T-001": {"STANDARD": ["Okay, I'll take the wrong bus."], "LOCAL": []},
    })
    settings = dict(lora_rank=2, lora_alpha=4, epochs=1, batch_size=4, max_source=32, max_target=12,
                    output_dir=tmp_path / "artifacts")

    report = finetune(base, train, validation, negatives_path=candidates, unlikelihood_weight=0.5, **settings)

    assert report["unlikelihood"]["negatives"] == 1 and report["unlikelihood"]["weight"] == 0.5
    assert report["history"][1]["unlikelihood_loss"] >= 0  # an untrained tiny model gives the bad reply almost no chance
    assert finetune(base, train, validation, output_name="plain", **settings)["unlikelihood"] is None
    with pytest.raises(ValueError, match="must be positive"):
        finetune(base, train, validation, output_name="bad", negatives_path=candidates, unlikelihood_weight=0,
                 **settings)


def write_preferences(path: Path, questions: list[tuple[str, str, list[str], str | None]]) -> Path:
    """A compare_v5 make-prefs sheet: (item id, style, [reply 1, reply 2], answer) per question."""
    path.write_text(json.dumps({"questions": [
        {"item_id": item_id, "style": style, "replies": replies, "answer": answer}
        for item_id, style, replies, answer in questions
    ]}), encoding="utf-8")
    return path


PREFERENCES = [
    ("ITEM-T-001", "STANDARD", ["Okay, I'll take the wrong bus.", "No worries, tell me when you reach."], "second"),
    ("ITEM-T-001", "LOCAL", ["no worries lah", "ok i take wrong bus also"], "first"),
    ("ITEM-T-002", "STANDARD", ["okay, i'll order first", "That's fine"], "tie"),
    ("ITEM-T-002", "LOCAL", ["ok i order later", "ok"], "both_bad"),
    ("ITEM-T-002", "LOCAL", ["ok i order later", "ok"], None),
]


def test_preference_pairs_turn_picks_into_better_and_worse_rows(tmp_path: Path):
    from replycoach.generate.finetune_v5 import preference_pairs

    train = item_rows(write_items(tmp_path / "train.jsonl", ECHO_ITEMS))

    pairs, unused, from_both_bad = preference_pairs(write_preferences(tmp_path / "prefs.json", PREFERENCES), train)

    assert unused == 1  # the tie; the unanswered question counts as nothing
    team_local = next(row["response"] for row in train if row["item_id"] == "ITEM-T-002" and row["style"] == "LOCAL")
    assert [(better["response"], worse["response"]) for better, worse in pairs] == [
        ("No worries, tell me when you reach.", "Okay, I'll take the wrong bus."),
        ("no worries lah", "ok i take wrong bus also"),
        # "both bad": the team's own reply beats each of the two, except a copy of itself.
        (team_local, "ok")]
    assert team_local == "ok i order later" and from_both_bad == 1
    assert pairs[0][0]["message"] == pairs[0][1]["message"] == "sorry, I took the wrong bus"
    outside = write_preferences(tmp_path / "outside.json", [("ITEM-V-001", "LOCAL", ["a", "b"], "first")])
    with pytest.raises(ValueError, match="outside the training file"):
        preference_pairs(outside, train)


def test_dpo_loss_falls_as_the_better_reply_gains_on_the_worse(monkeypatch):
    import torch

    from replycoach.generate import finetune_v5

    reference = torch.tensor([[-5.0, -5.0]])
    losses = []
    for better, worse in ((-5.0, -5.0), (-3.0, -7.0), (-7.0, -3.0)):
        monkeypatch.setattr(finetune_v5, "sequence_logprobs",
                            lambda model, encoded, pad_id, device, b=better, w=worse: torch.tensor([b, w]))
        loss, agreement = finetune_v5.dpo_loss(None, [([1], [2])], reference, 0, "cpu", beta=0.5)
        losses.append((round(float(loss), 4), agreement))

    # At the start the loss is log 2; it falls when the better reply gains and rises when the worse one does.
    assert losses[0] == (round(float(torch.log(torch.tensor(2.0))), 4), 0.0)
    assert losses[1][0] < losses[0][0] < losses[2][0]
    assert (losses[1][1], losses[2][1]) == (1.0, 0.0)


def test_dpo_training_reports_its_pairs_and_agreement(tmp_path: Path):
    texts = [text for memory, message, standard, local in ECHO_ITEMS for text in [message, standard, local]]
    replies = [text for _, _, pair, _ in PREFERENCES for text in pair]
    base = tiny_model(tmp_path / "tiny", [*item_texts(), *texts, *replies])
    train = write_items(tmp_path / "train.jsonl", ECHO_ITEMS)
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[3:], prefix="ITEM-V")
    prefs = write_preferences(tmp_path / "prefs.json", PREFERENCES)
    settings = dict(lora_rank=2, lora_alpha=4, epochs=1, batch_size=4, max_source=32, max_target=12,
                    output_dir=tmp_path / "artifacts")

    report = finetune(base, train, validation, preferences_path=prefs, dpo_weight=0.5, **settings)

    assert report["preferences"] == {"sheet": str(prefs), "pairs": 3, "from_both_bad": 1, "unused_answers": 1,
                                     "weight": 0.5, "beta": 0.1}
    assert report["history"][1]["dpo_loss"] > 0 and 0 <= report["history"][1]["dpo_agreement"] <= 1
    assert finetune(base, train, validation, output_name="plain", **settings)["preferences"] is None
    unanswered = write_preferences(tmp_path / "none.json", [PREFERENCES[2], PREFERENCES[4]])
    with pytest.raises(ValueError, match="no questions answered"):
        finetune(base, train, validation, output_name="empty", preferences_path=unanswered, **settings)


def test_joint_rows_hold_both_styles_standard_first(tmp_path: Path):
    from replycoach.generate.finetune_v5 import joint_rows

    rows = joint_rows(item_rows(write_items(tmp_path / "train.jsonl", ITEMS[:2])))

    assert [(row["item_id"], row["style"], row["response"]) for row in rows] == [
        ("ITEM-T-001", "BOTH", "Sure, count me in! | LOCAL: steady lah"),
        ("ITEM-T-002", "BOTH", "I am on my way now. | LOCAL: otw liao")]
    assert rows[0]["context"] == ITEMS[0][0]


def test_repeats_are_blocked_within_each_half_of_a_joint_reply():
    import torch

    from replycoach.generate.finetune_v5 import NoRepeatWithinHalf

    blocker = NoRepeatWithinHalf(3, marker=[8, 9])  # 8 9 stands for "LOCAL:"
    standard = [0, 5, 6, 7, 1]  # start token, then "i wait for you"

    def banned(ids: list[int]) -> list[int]:
        return blocker(torch.tensor([ids]), torch.zeros(1, 10))[0].isinf().nonzero().flatten().tolist()

    assert banned(standard + [8, 9, 5, 6]) == []  # the LOCAL half may say "i wait for" again
    assert banned(standard + [8, 9, 5, 6, 7, 5, 6]) == [7]  # but not twice within itself
    assert banned([0, 5, 6, 7, 5, 6]) == [7]  # before the separator the STANDARD half is checked the same way
    assert banned([0, 5, 6, 7, 3, 3]) == []


def test_joint_training_writes_both_styles_in_one_reply(tmp_path: Path):
    from replycoach.generate.decode_v5 import decode
    from replycoach.generate.finetune_v5 import input_format

    base = tiny_model(tmp_path / "tiny", [*item_texts(), "BOTH", "|", "LOCAL:"])
    train = write_items(tmp_path / "train.jsonl", ITEMS[:3])
    validation = write_items(tmp_path / "validation.jsonl", ITEMS[3:], prefix="ITEM-V")
    settings = dict(lora_rank=2, lora_alpha=4, epochs=1, batch_size=4, max_source=32, max_target=12,
                    output_dir=tmp_path / "artifacts")

    report = finetune(base, train, validation, output_name="joint", joint=True, **settings)

    assert report["joint"] is True and report["train_items"] == 3
    assert set(report["history"][1]) >= {"validation_loss", "validation_loss_both"}
    assert input_format(tmp_path / "artifacts" / "joint" / "best")["joint"] is True
    [output] = report["validation_outputs"]
    assert {"generated_standard", "generated_local", "joint_reply", "joint_split"} <= set(output)
    assert report["joint_unsplit"] == (not output["joint_split"])
    decoded = decode(tmp_path / "artifacts" / "joint" / "best", "joint", {"num_beams": 1}, validation,
                     max_source=32, max_target=12, output_dir=tmp_path / "artifacts")
    assert decoded["joint"] is True and "joint_reply" in decoded["validation_outputs"][0]
    with pytest.raises(ValueError, match="does not take"):
        finetune(base, train, validation, output_name="mixed", joint=True,
                 preferences_path=tmp_path / "prefs.json", **settings)


def test_optimizer_step_budget_stops_mid_epoch_and_saves_validation(tmp_path):
    base = tiny_model(tmp_path / 'tiny', item_texts())
    train = write_items(tmp_path / 'train.jsonl', ITEMS[:3])
    val = write_items(tmp_path / 'validation.jsonl', ITEMS[3:], prefix='ITEM-V')
    report = finetune(base,train,val,lora_rank=2,lora_alpha=4,epochs=5,batch_size=2,
                      max_source=32,max_target=12,max_steps=2,
                      output_name='budget',output_dir=tmp_path/'artifacts')
    assert report['optimizer_steps']==2
    assert report['stopped_at_step_budget']
    assert [h['epoch'] for h in report['history']]==[0,1]
    assert report['history'][-1]['optimizer_steps']==2
    assert len(report['validation_outputs'])==1
    assert (tmp_path/'artifacts/budget/best/adapter_config.json').exists()


@pytest.mark.parametrize('budget',[0,-1,True,1.5])
def test_invalid_optimizer_step_budget_rejected_before_model_load(tmp_path,budget):
    with pytest.raises(ValueError,match='max_steps'):
        finetune('unloaded',max_steps=budget,output_dir=tmp_path)


def test_standard_emphasis_changes_only_training_exposure(tmp_path):
    base = tiny_model(tmp_path / 'tiny', item_texts())
    train = write_items(tmp_path / 'train.jsonl', ITEMS[:3])
    validation = write_items(tmp_path / 'validation.jsonl', ITEMS[3:], prefix='ITEM-V')
    report = finetune(base, train, validation, output_dir=tmp_path / 'artifacts', output_name='weighted',
                      lora_rank=2, lora_alpha=4, epochs=1, batch_size=2, max_source=64, max_target=12,
                      rewrite_local=True, standard_repeats=2, save_final=True)
    assert report['train_pairs'] == 9
    assert report['standard_extra_rows'] == 3
    assert report['train_items'] == 3
    assert report['validation_items'] == 1
    assert len(report['final_validation_outputs']) == 1
    assert report['final_validation_outputs'][0]['generated_standard'] is not None


def test_step_snapshots_do_not_change_training_trajectory(tmp_path):
    import torch
    from safetensors.torch import load_file
    base = tiny_model(tmp_path / 'tiny', item_texts())
    train = write_items(tmp_path / 'train.jsonl', ITEMS[:3])
    val = write_items(tmp_path / 'validation.jsonl', ITEMS[3:], prefix='ITEM-V')
    settings = dict(lora_rank=2, lora_alpha=4, epochs=3, batch_size=2, max_source=64,
                    max_target=12, max_steps=3, save_final=True, rewrite_local=True,
                    output_dir=tmp_path / 'artifacts')
    plain = finetune(base, train, val, output_name='plain', **settings)
    captured = finetune(base, train, val, output_name='captured', save_steps=(1, 3), **settings)
    assert set(captured['step_checkpoints']) == {'1', '3'}
    def weights(name):
        return load_file(str(tmp_path / 'artifacts' / name / 'adapter_model.safetensors'))
    expected = weights('plain/final')
    for name in ['captured/final', 'captured/step_3']:
        actual = weights(name)
        assert all(torch.equal(v, actual[k]) for k, v in expected.items())
    assert any(not torch.equal(v, weights('captured/step_1')[k]) for k, v in expected.items())
    assert plain['final_validation_outputs'] == captured['final_validation_outputs']
    metadata = json.loads((tmp_path / 'artifacts/captured/step_1/v5_input.json').read_text())
    assert metadata['rewrite_local'] and metadata['max_source'] == 64


@pytest.mark.parametrize('steps', [(0,), (True,), (1, 1), (4,)])
def test_invalid_step_snapshots_fail_before_model_load(tmp_path, steps):
    with pytest.raises(ValueError, match='save_steps'):
        finetune('unloaded', output_dir=tmp_path, max_steps=3, save_steps=steps)
