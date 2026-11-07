"""Package existing development outputs for user review; performs no inference."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/component_baseline_v1/style_review"
SELECTED = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 22, 24]
NOTES = {
    1: "Timing preserved. Longer and more formal than many original LOCAL examples.",
    2: "Departure and one-hour conflict preserved. 'already' and 'liao' together may feel overdone; user judgment needed.",
    3: "Unavailability and Thursday alternative preserved; resembles original terse refusal style.",
    5: "Return direction and deadline preserved. Desk is an optional detail introduced by STANDARD, not the rewriter. Mostly English wording is not automatically a defect.",
    6: "Permission retained but 'after class' is implicit; repeated 'Can' is a style question, not an ownership reversal.",
    7: "Acceptance retained. 'Help me a lot leh' may sound forced; user judgment needed.",
    8: "Sender and tonight deadline preserved; concise wording resembles original examples.",
    9: "Completed upload and recent timing preserved. Check whether repeated completion cues feel natural.",
    10: "Not-started state preserved; colloquial grammar resembles original examples.",
    11: "Exact room retained. 'already' adds an aspect marker without changing the stated current location.",
    12: "Colour preserved. 'one' is optional; user judgment needed on whether it sounds natural here.",
    13: "Uncertainty and checking retained; no fabricated closing time.",
    15: "Eight-dollar budget and refusal preserved. Wording is longer than necessary; no demonstrated contradiction.",
    22: "Black-coffee choice and recipient retained; no buyer reversal.",
    24: "STANDARD and LOCAL both omit clarification of unspecified location. 'head over' becomes 'go over straight away', slightly stronger immediacy.",
}


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {name: f"artifacts/component_baseline_v1/{name}.json" for name in ("gemma12b", "qwen4b", "mt5")}
    reports = {name: {c["item_id"]: c for c in read(path)["cases"]} for name, path in paths.items()}
    inputs = {c["item_id"]: c for c in read("data/evaluation/component_baseline_v1/inputs.json")}
    rows = []
    for number in SELECTED:
        item_id = f"COMP1-{number:03d}"
        c = inputs[item_id]
        g, q, m = (reports[n][item_id] for n in ("gemma12b", "qwen4b", "mt5"))
        assert g["direct_local"]["standard_anchor"] == g["direct_standard"]["value"]["reply"]
        rows.append({
            "item_id": item_id,
            "context": " | ".join(f"{s}: {t}" for s, t in zip(c["speakers"], c["context"], strict=True)),
            "latest_peer_message": c["message"],
            "gemma_standard": g["direct_standard"]["value"]["reply"],
            "gemma_local": g["direct_local"]["value"]["reply"],
            "qwen_direct_local": q["direct_local"]["value"]["reply"],
            "mt5_direct_local": m["direct_local"]["reply"],
            "assistant_observation": NOTES[number],
            "user_would_send_yes_no_edit": "",
            "user_meaning_preserved_yes_no_unsure": "",
            "user_naturalness_1_to_5": "",
            "user_preferred_wording_optional": "",
        })
    with (OUT / "review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_path = ROOT / "data/dev/train.jsonl"
    train = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    # Fixed row positions, chosen without generated-output matching or test access.
    originals = [{"item_id": train[i]["id"], **train[i]["replies"]} for i in [4, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]]
    original_ids = {r["id"] for r in train}
    exemplars = read("data/evaluation/component_baseline_v1/style_examples.json")
    assert all(r["item_id"] in original_ids for r in exemplars)
    packet = {
        "scope": "15 deliberately selected development cases; diagnostic selection, not an unbiased quality estimate. User fields remain blank. Assistant observations are exploratory, not human ratings.",
        "selection": "Timing, availability, ownership, sender direction, completion, factual answers, uncertainty, budget, preference, and missing-location clarification. Excludes speaker-ambiguous emotional cases 017–020.",
        "original_style_examples_seen_by_models": exemplars,
        "additional_original_training_style_examples": originals,
        "reference_limit": "Original pairs show varied author styles and sometimes content differences. They illustrate wording, not flawless semantic targets. Additional originals were not added to inference prompts.",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [*paths.values(), "data/dev/train.jsonl", "data/evaluation/component_baseline_v1/inputs.json"]},
        "findings": [
            "Across these 15 inspected Gemma direct pairs, no clear reversal of colour, negation, ownership, or sender was identified. This is a limited assistant judgment, not a measured reliability rate.",
            "The missing-location reply (024) needs contextual improvement; rewriting cannot recover a question STANDARD never asked.",
            "Original LOCAL examples range from near-English wording to contractions, abbreviations and particles. A particle on every sentence is not a target supported by this small reference set.",
            "Gemma has useful concise rewrites (003, 008, 010, 013), but optional particles, repetition and long refusals merit user review (002, 006, 007, 012, 015).",
        ],
        "next_decision": "Collect user preferences before changing style prompts or training. Preserve meaning and original diverse styles; do not convert assistant observations into human labels.",
        "cases": rows,
    }
    (OUT / "review.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    with (OUT / "original_style_examples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "standard", "local"])
        writer.writeheader()
        writer.writerows(originals)
    with (OUT / "review.csv").open(newline="") as f:
        restored = list(csv.DictReader(f))
    assert restored == rows
    assert len(rows) == len({r["item_id"] for r in rows}) == 15
    assert all(not r[k] for r in rows for k in r if k.startswith("user_"))
    print(f"Verified {len(rows)} matched cases, 12 additional original style examples, CSV round trip, blank user ratings, and exact Gemma rewrite anchors.")


if __name__ == "__main__":
    main()
