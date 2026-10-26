"""Convert DailyDialog-style utterance lists into (context, response) pairs later.

This script does not download data. Point it at a local JSONL of
{"utterances": ["...", "..."]} after Role C obtains DailyDialog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replycoach.tokenize import tokenize


def dialogue_key(utterances: list[str]) -> tuple[str, ...]:
    return tuple(" ".join(tokenize(turn)) for turn in utterances)


def pairs_from_utterances(utterances: list[str], max_context: int = 3) -> list[dict]:
    rows = []
    for idx in range(1, len(utterances)):
        context = utterances[max(0, idx - max_context) : idx]
        rows.append(
            {
                "context": context,
                "message": utterances[idx - 1] if False else context[-1],
                "response": utterances[idx],
                "source": "dailydialog",
            }
        )
    # message is the last context turn; response is the next turn.
    cleaned = []
    for row in rows:
        cleaned.append(
            {
                "context": row["context"][:-1],
                "message": row["context"][-1],
                "response": row["response"],
                "source": "dailydialog",
            }
        )
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-dialogues", type=Path,
                        help="Optional local validation utterance JSONL; remove matching complete TRAIN dialogues.")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or (args.exclude_dialogues and args.exclude_dialogues.resolve() == args.output.resolve()):
        parser.error("output must not replace source files")
    excluded = set()
    if args.exclude_dialogues:
        with args.exclude_dialogues.open(encoding="utf-8") as handle:
            excluded = {dialogue_key(json.loads(line)["utterances"]) for line in handle if line.strip()}
    count = 0
    skipped = 0
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            utterances = json.loads(line)["utterances"]
            if dialogue_key(utterances) in excluded:
                skipped += 1
                continue
            for row in pairs_from_utterances(utterances):
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} pairs to {args.output}")
    if args.exclude_dialogues:
        print(f"excluded {skipped} training dialogues matching validation")


if __name__ == "__main__":
    main()
