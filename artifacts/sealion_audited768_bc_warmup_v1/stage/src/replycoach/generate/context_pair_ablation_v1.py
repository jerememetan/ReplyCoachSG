"""Matched main-example exposures and an isolated context-pair objective."""

import hashlib
import json
import random

from replycoach.generate.mt5_matched_v1 import paired_rows, source, task_rows


def extract_context_pairs(records):
    result = []
    for item in records:
        turns = item["session_memory"]
        for i in range(1, len(turns)):
            if turns[i]["speaker"] == "user" and turns[i - 1]["speaker"] != "user":
                result.append(dict(item_id=f"{item['id']}-CTX-{i}", origin_item_id=item["id"],
                                   style="REPLY", message=turns[i - 1]["text"],
                                   context=[t["text"] for t in turns[:i - 1]],
                                   speakers=[t["speaker"] for t in turns[:i - 1]], response=turns[i]["text"]))
    return result


def main_rows(records, eligible):
    standard = [r for r in task_rows(records) if r["style"] == "STANDARD"]
    rewrite = [dict(r, style="LOCAL_REWRITE") for r in paired_rows(records, eligible)]
    return standard + rewrite


def encode_row(row, tokenizer):
    text = source(row, row["style"], row.get("standard_anchor"))
    inputs = tokenizer(text, add_special_tokens=True)["input_ids"]
    targets = tokenizer(row["response"], add_special_tokens=True)["input_ids"]
    if not row["response"].strip() or len(inputs) > 192 or len(targets) > 48:
        raise ValueError("Empty or over-budget row: " + row["item_id"])
    return inputs, targets


def schedule(main_count, auxiliary_count, seed, steps=784, main_per_step=15):
    if min(main_count, auxiliary_count, steps, main_per_step) < 1:
        raise ValueError("Positive schedule dimensions required")

    def stream(count, offset):
        cycle = 0
        while True:
            order = list(range(count))
            random.Random(seed + offset + cycle).shuffle(order)
            yield from order
            cycle += 1

    primary, auxiliary = stream(main_count, 0), stream(auxiliary_count, 1_000_000)
    return [dict(main=[next(primary) for _ in range(main_per_step)], auxiliary=next(auxiliary)) for _ in range(steps)]


def schedule_sha256(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def backward_auxiliary(loss, weight):
    """Execute the same graph in both arms; zero weight sends no gradients."""
    if weight < 0:
        raise ValueError("Nonnegative auxiliary coefficient required")
    if not loss.isfinite():
        raise FloatingPointError("Nonfinite auxiliary loss")
    (loss * weight).backward()
