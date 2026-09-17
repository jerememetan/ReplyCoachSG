"""Week 7 helper: compare retrieved vs generated candidates on development items only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replycoach.io import load_items
from replycoach.pipeline import ReplyCoach
from replycoach.retrieval.v0 import V0Retriever
from replycoach.tokenize import tokenize


def token_f1(pred: str, gold: str) -> float:
    p, g = set(tokenize(pred)), set(tokenize(gold))
    if not p or not g:
        return 0.0
    overlap = len(p & g)
    precision = overlap / len(p)
    recall = overlap / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None, help="Development JSONL. Never pass the locked test set.")
    args = parser.parse_args()
    items = load_items(args.data) if args.data else load_items()
    coach = ReplyCoach(retriever=V0Retriever.from_items(items), items=items)
    rows = []
    for item in items:
        if item.split == "test":
            raise SystemExit("refusing to score the locked test split")
        result = coach.suggest(item.message.text, session=item.memory_texts(), log=False)
        rows.append(
            {
                "id": item.id,
                "standard_f1": round(token_f1(result.standard.candidate_text, item.replies.standard), 3),
                "local_f1": round(token_f1(result.local.candidate_text, item.replies.local), 3),
                "standard_source": result.standard.source,
                "local_source": result.local.source,
            }
        )
    mean_s = sum(r["standard_f1"] for r in rows) / max(len(rows), 1)
    mean_l = sum(r["local_f1"] for r in rows) / max(len(rows), 1)
    print(json.dumps({"n": len(rows), "mean_standard_f1": round(mean_s, 3), "mean_local_f1": round(mean_l, 3), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
