import json
from pathlib import Path

import torch

from replycoach.generate.context_pair_ablation_v1 import backward_auxiliary, extract_context_pairs, main_rows, schedule, schedule_sha256
from replycoach.generate.finetune_v5 import context_pairs
from replycoach.generate.mt5_matched_v1 import source

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_context_pairs_exactly_match_historical_extraction_without_changing_team_data():
    path = ROOT / "data/dev/train.jsonl"
    before = path.read_bytes()
    current = extract_context_pairs(records(path))
    historical = context_pairs(path)
    assert len(current) == len(historical) == 180
    for new, old in zip(current, historical, strict=True):
        assert new["origin_item_id"] == old["item_id"]
        assert {k: new[k] for k in ["style", "message", "context", "speakers", "response"]} == {k: old[k] for k in ["style", "message", "context", "speakers", "response"]}
    assert path.read_bytes() == before


def test_main_examples_and_masks_share_one_reproducible_budget():
    a = schedule(1535, 180, 1)
    b = schedule(1535, 180, 1)
    assert a == b and schedule_sha256(a) == schedule_sha256(b)
    assert a != schedule(1535, 180, 2)
    assert len(a) == 784 and sum(len(x["main"]) for x in a) == 11760
    first_cycle = [j for step in a for j in step["main"]][:1535]
    assert len(set(first_cycle)) == 1535
    assert {step["auxiliary"] for step in a[:180]} == set(range(180))


def test_zero_auxiliary_weight_cannot_change_existing_primary_gradients():
    model = torch.nn.Linear(3, 2)
    primary = model(torch.ones(1, 3)).square().mean()
    primary.backward()
    before = [p.grad.clone() for p in model.parameters()]
    backward_auxiliary(model(torch.tensor([[1., 2., 3.]])).square().mean(), 0.0)
    assert all(torch.equal(old, p.grad) for old, p in zip(before, model.parameters(), strict=True))
    backward_auxiliary(model(torch.tensor([[1., 2., 3.]])).square().mean(), 180 / 1535)
    assert any(not torch.equal(old, p.grad) for old, p in zip(before, model.parameters(), strict=True))


def test_mixed_rows_keep_targets_out_of_inputs_and_eligible_rewrite_pairs_only():
    train = records(ROOT / "data/training_expansion/generated600_repair_v1/combined_train.jsonl")
    audit = json.loads((ROOT / "data/training_expansion/sealion_rewrite_v1/pair_audit.json").read_text())
    eligible = {x["id"] for x in audit["items"] if x["split"] == "train" and x["eligible"]}
    rows = main_rows(train, eligible)
    assert len(rows) == 1535
    assert sum(r["style"] == "STANDARD" for r in rows) == 790
    assert sum(r["style"] == "LOCAL_REWRITE" for r in rows) == 745
    row = next(r for r in rows if r["style"] == "LOCAL_REWRITE")
    changed = dict(row, response="SECRET LOCAL TRAINING TARGET")
    assert source(row, row["style"], row["standard_anchor"]) == source(changed, changed["style"], changed["standard_anchor"])
    assert "SECRET LOCAL TRAINING TARGET" not in source(changed, changed["style"], changed["standard_anchor"])
