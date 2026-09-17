"""Week 4 Friday smoke test: same data format, V0 retrieves a reply, V3 trains one batch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from replycoach.generate.train_v3 import run_one_batch
from replycoach.io import load_items
from replycoach.pipeline import ReplyCoach
from replycoach.schema import Candidate


def main() -> None:
    items = load_items()
    print(f"loaded {len(items)} items from shared JSONL")
    for item in items:
        item.validate()
    print("schema: all items valid")

    coach = ReplyCoach.from_pilot()
    result = coach.suggest(
        "eh later go makan? can 吃饭 then library",
        session=["lecture ending soon sia", "yeah I'm starving"],
    )
    Candidate.from_dict(result.standard.to_dict())
    Candidate.from_dict(result.local.to_dict())
    print("retrieval demo:")
    print(f"  STANDARD: {result.standard.candidate_text}")
    print(f"  LOCAL:    {result.local.candidate_text}")
    print(f"  detected: {[t['surface'] for t in result.detected_terms]}")

    try:
        report = run_one_batch(batch_size=4, use_attention=False)
        print("rnn one-batch:", json.dumps(report))
    except ImportError as exc:
        print("rnn one-batch skipped (install torch):", exc)
        report = {"skipped": True}

    summary = {
        "items": len(items),
        "standard": result.standard.candidate_text,
        "local": result.local.candidate_text,
        "rnn": report,
    }
    print("WEEK4_OK")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
