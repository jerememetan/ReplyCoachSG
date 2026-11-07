"""Offline checks for training-only critic labels and DPO compatibility."""
import copy
import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def critic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("critic_preferences")


def question(critic):
    item = critic.load_items(critic.ROOT / "data/dev/train.jsonl")[0]
    return {"item_id": item.id, "style": "LOCAL", "answer": None,
            "conversation": [[t.speaker, t.text] for t in item.session_memory],
            "message": item.message.text, "replies": ["first reply", "second reply"]}


def decision(choice, uncertain=False):
    return {"mapped_choice": choice, "uncertain": uncertain}


@pytest.mark.parametrize("a,b,expected", [
    (decision("first"), decision("first"), ("first", "accepted")),
    (decision("first"), decision("second"), (None, "order_disagreement")),
    (decision("first", True), decision("first"), (None, "uncertain")),
    (decision("both_bad"), decision("both_bad"), (None, "both_bad")),
    (decision("tie"), decision("tie"), (None, "tie")),
])
def test_consensus(critic, a, b, expected):
    assert critic.consensus(a, b) == expected


def test_training_identity_and_no_existing_labels(critic):
    q = question(critic)
    critic.validate_sheet({"questions": [q]})
    for field, value in [("item_id", "not-a-training-item"), ("message", "altered"), ("answer", "first")]:
        bad = {**q, field: value}
        with pytest.raises(ValueError):
            critic.validate_sheet({"questions": [bad]})


def test_swapped_choice_maps_to_original_order(critic, monkeypatch):
    tasks = []
    def request(base, route, payload):
        tasks.append(json.loads(payload["messages"][1]["content"]))
        return {"done": True, "message": {"content": json.dumps(
            {"choice": "A", "a_acceptable": True, "b_acceptable": False,
             "uncertain": False, "reason": "Better answer."})}}
    monkeypatch.setattr(critic, "request_json", request)
    settings = {"judge_model": "fake", "think": False, "options": {}}
    q = question(critic)
    assert critic.judge(q, {"base": "unused"}, settings)["mapped_choice"] == "first"
    assert critic.judge(q, {"base": "unused"}, settings, swap=True)["mapped_choice"] == "second"
    assert tasks[0]["A"] == tasks[1]["B"]
    assert "item_id" not in tasks[0]


def test_complete_sheet_is_consumable_by_dpo(critic, monkeypatch, tmp_path):
    pytest.importorskip("peft")
    from replycoach.generate.finetune_v5 import item_rows, preference_pairs

    q = question(critic)
    source, output = tmp_path / "source.json", tmp_path / "labels.json"
    source.write_text(json.dumps({"questions": [q]}))
    monkeypatch.setattr(sys, "argv", ["critic_preferences", str(source), "--output", str(output)])
    monkeypatch.setattr(critic, "prepare_judge", lambda _: {"judge_model": "fake", "model_digest": "digest"})
    monkeypatch.setattr(critic, "judge", lambda *args, **kwargs: decision("second"))
    critic.main()
    sheet = json.loads(output.read_text())
    assert sheet["training_critic"]["status"] == "complete"
    rows = item_rows(critic.ROOT / "data/dev/train.jsonl")
    pairs, _, _ = preference_pairs(output, rows)
    assert pairs[0][0]["response"] == "second reply"
    assert pairs[0][1]["response"] == "first reply"
    older = copy.deepcopy(sheet)
    older["training_critic"].pop("prompt_version")
    output.write_text(json.dumps(older))
    with pytest.raises(ValueError, match="prompt version"):
        preference_pairs(output, rows)
    sheet["training_critic"]["status"] = "failed"
    output.write_text(json.dumps(sheet))
    with pytest.raises(ValueError, match="complete"):
        preference_pairs(output, rows)


def test_calibration_flags_unsafe_acceptance(critic):
    calibration = importlib.import_module("calibrate_critic")
    result = calibration.assess(decision("first"), decision("first"), "both_bad")
    assert result == {"correct": False, "order_agreement": True, "unsafe_acceptance": True}
    assert calibration.assess(decision("both_bad", True), decision("both_bad", True), "both_bad")["correct"]


def test_calibration_checks_acceptability_even_when_ranking_correct(critic):
    calibration = importlib.import_module("calibrate_critic")
    first = {**decision("first"), "a_acceptable": True, "b_acceptable": True}
    swapped = {**decision("first"), "a_acceptable": True, "b_acceptable": True}
    result = calibration.assess(first, swapped, "first", [True, False])
    assert result["order_agreement"]
    assert not result["correct"]
    assert result["unsafe_acceptance"]
    assert result["acceptance_errors"] == 2


def test_calibration_allows_unknown_personal_status_tie(critic):
    calibration = importlib.import_module("calibrate_critic")
    result = {**decision("tie"), "a_acceptable": True, "b_acceptable": True}
    assessment = calibration.assess(result, result, "tie", [True, True])
    assert assessment["correct"]
    assert not assessment["unsafe_acceptance"]


@pytest.mark.parametrize('verdicts,expected', [
    (['acceptable','unacceptable'], ('first','accepted')),
    (['unacceptable','acceptable'], (None,'screen_rejected')),
    (['acceptable','uncertain'], (None,'screen_uncertain')),
    (['acceptable','acceptable'], ('first','accepted')),
])
def test_isolated_screen_gates_selected_preference(critic, verdicts, expected):
    isolated = [{'verdict':v} for v in verdicts]
    assert critic.screened_consensus(decision('first'), decision('first'), isolated) == expected
    assert critic.screened_consensus(decision('first'), decision('second'), isolated) == (None,'order_disagreement')


@pytest.mark.parametrize('verdict,answer,status', [
    ('acceptable','first','accepted'), ('unacceptable',None,'screen_rejected'),
])
def test_screened_pilot_records_decisions_and_skips_rejected_labels(critic, monkeypatch, tmp_path, verdict, answer, status):
    modes = importlib.import_module('compare_critic_modes')
    q = question(critic)
    source, output = tmp_path/'source.json', tmp_path/'screened.json'
    source.write_text(json.dumps({'questions':[q]}))
    monkeypatch.setattr(sys,'argv',['critic',str(source),'--output',str(output),'--screen-single'])
    monkeypatch.setattr(critic,'prepare_judge',lambda _: {'judge_model':'fake','model_digest':'fixed'})
    monkeypatch.setattr(critic,'judge',lambda *args,**kwargs: decision('first'))
    monkeypatch.setattr(modes,'judge_single',lambda *args: {'reason':'Fixture.', 'verdict':verdict})
    critic.main()
    sheet=json.loads(output.read_text())
    assert sheet['training_critic']['acceptability_screen']['prompt_version'] == modes.SINGLE_VERSION
    assert sheet['training_critic']['counts'] == {status:1}
    assert sheet['questions'][0]['answer'] == answer
    assert len(sheet['questions'][0]['critic']['single']) == 2
