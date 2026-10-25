import json
from pathlib import Path

import pytest

from replycoach.generate.compare_v5 import make_preference_questions, make_questions, rate, score, sign_test

ITEMS = [(["are you free later"], "i took the wrong bus"), ([], "want to eat?"), (["hello"], "wya")]


def write_items(path: Path, split: str = "dev") -> Path:
    rows = [{"id": f"ITEM-V-{index:03d}", "split": split,
             "session_memory": [{"speaker": "user", "text": text} for text in memory],
             "message": {"text": message}, "replies": {"standard": "ok", "local": "ok lor"}}
            for index, (memory, message) in enumerate(ITEMS, start=1)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def run(standard: list[str], local: list[str]) -> dict:
    return {"validation_outputs": [
        {"item_id": f"ITEM-V-{index:03d}", "generated_standard": s, "generated_local": l}
        for index, (s, l) in enumerate(zip(standard, local), start=1)
    ]}


RUN_A = run(["no worries", "sure", "at the mrt"], ["no worries lah", "can", "mrt"])
RUN_B = run(["ok see you later", "sure", "omw"], ["ok see you later", "can lah", "mrt"])


def sheet(questions: list[dict], same: dict) -> dict:
    return {"runs": ["a", "b"], "same_reply": same, "questions": questions}


def test_same_replies_are_skipped_and_the_rest_shuffled(tmp_path: Path):
    questions, same = make_questions(RUN_A, RUN_B, write_items(tmp_path / "v.jsonl"))

    assert same == {"ALL": 2, "STANDARD": 1, "LOCAL": 1}
    assert len(questions) == 4
    for question in questions:
        style = f"generated_{question['style'].lower()}"
        a, b = (next(row[style] for row in r["validation_outputs"] if row["item_id"] == question["item_id"])
                for r in (RUN_A, RUN_B))
        assert question["replies"] == ([a, b] if question["a_first"] else [b, a])
    assert make_questions(RUN_A, RUN_B, tmp_path / "v.jsonl")[0] == questions  # the seed fixes the order


def test_only_development_items_and_matching_runs(tmp_path: Path):
    with pytest.raises(ValueError, match="development"):
        make_questions(RUN_A, RUN_B, write_items(tmp_path / "t.jsonl", split="test"))
    with pytest.raises(ValueError, match="same development items"):
        make_questions(RUN_A, run(["x"], ["y"]), write_items(tmp_path / "v.jsonl"))


def test_rating_saves_each_answer_and_resumes(tmp_path: Path):
    questions, same = make_questions(RUN_A, RUN_B, write_items(tmp_path / "v.jsonl"))
    path = tmp_path / "ab.json"
    answers = iter(["huh", "1", "=", "q"])
    shown = []

    rate(sheet(questions, same), path, ask=lambda prompt: next(answers), say=shown.append)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [question["answer"] for question in saved["questions"]] == ["first", "tie", None, None]
    assert "  Friend: " in shown[0] and "  1) " in shown[0] and "  2) " in shown[0]
    rate(saved, path, ask=lambda prompt: "x", say=shown.append)
    assert [question["answer"] for question in saved["questions"]] == ["first", "tie", "both_bad", "both_bad"]


def test_score_reveals_which_run_won():
    questions = [{"style": "STANDARD", "a_first": True, "answer": "first"},
                 {"style": "STANDARD", "a_first": False, "answer": "first"},
                 {"style": "LOCAL", "a_first": False, "answer": "second"},
                 {"style": "LOCAL", "a_first": True, "answer": "tie"},
                 {"style": "LOCAL", "a_first": True, "answer": None}]

    result = score(sheet(questions, {"ALL": 3, "STANDARD": 1, "LOCAL": 2}))

    assert {key: result["ALL"][key] for key in ("a", "b", "tie", "same_reply")} == {"a": 2, "b": 1, "tie": 1,
                                                                                     "same_reply": 3}
    assert (result["STANDARD"]["a"], result["STANDARD"]["b"]) == (1, 1)
    assert (result["LOCAL"]["a"], result["LOCAL"]["b"], result["LOCAL"]["tie"]) == (1, 0, 1)
    assert (result["answered"], result["questions"]) == (4, 5)


def test_sign_test():
    assert sign_test(0, 0) == 1.0
    assert sign_test(5, 5) == 1.0
    assert sign_test(10, 0) == pytest.approx(2 / 1024)
    assert sign_test(15, 5) == sign_test(5, 15) == pytest.approx(0.0414, abs=1e-4)


def test_preference_questions_put_the_usual_reply_against_a_sample(tmp_path: Path):
    items = write_items(tmp_path / "train.jsonl")
    candidates = {"items": [
        {"item_id": "ITEM-V-001", "candidates": {
            "STANDARD": [{"text": "ok see you", "beam": True}, {"text": "no worries", "beam": False},
                         {"text": "where are you", "beam": False}],
            "LOCAL": [{"text": "ok lor", "beam": True}]}},  # nothing to compare it with
        {"item_id": "ITEM-V-002", "candidates": {
            "STANDARD": [{"text": "sure", "beam": False}],  # no usual reply
            "LOCAL": [{"text": "can", "beam": True}, {"text": "can lah", "beam": False}]}},
    ]}

    questions = make_preference_questions(candidates, items)

    assert sorted((question["item_id"], question["style"]) for question in questions) == [
        ("ITEM-V-001", "STANDARD"), ("ITEM-V-002", "LOCAL")]
    for question in questions:
        usual = question["replies"][0 if question["a_first"] else 1]
        assert usual in ("ok see you", "can")
    with pytest.raises(ValueError, match="outside"):
        make_preference_questions({"items": [{"item_id": "ITEM-X-009", "candidates": {}}]}, items)
