"""Validate explicitly reviewed, paired synthetic examples before stage 2 training.

No API calls and no test-set reads. A review flag is an attestation, not an
automatic claim that a model's two replies preserve meaning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from replycoach.io import read_jsonl


def fingerprint(context: list[str], message: str) -> str:
    text = " ".join([*context, message]).casefold()
    return re.sub(r"\W+", " ", text).strip()


def load_synthetic(path: Path, protected_rows: list[dict]) -> list[dict]:
    """Return both styles together; reject malformed, duplicate or unreviewed items.

    Exact normalized conversation and message overlap is blocked. Paraphrase
    leakage still needs review; this does not establish semantic equivalence.
    """
    seen = {fingerprint(row["context"], row["message"]) for row in protected_rows}
    messages = {fingerprint([], row["message"]) for row in protected_rows}
    ids = {row["item_id"] for row in protected_rows}
    rows = []
    for number, item in enumerate(read_jsonl(path), 1):
        def require(condition, reason):
            if not condition:
                raise ValueError(f"{path}:{number}: {reason}")

        require(isinstance(item, dict), "expected an object")
        item_id = item.get("id")
        require(isinstance(item_id, str) and item_id.startswith("SYN-") and item_id not in ids,
                "unique SYN- id required")
        ids.add(item_id)
        provenance = item.get("provenance", {})
        require(isinstance(provenance, dict) and all(isinstance(provenance.get(k), str) and provenance[k].strip()
                for k in ("generator", "prompt_version", "batch_id")), "generation provenance required")
        review = item.get("review", {})
        require(isinstance(review, dict) and all(review.get(k) is True for k in
                ("meaning_preserved", "speaker_correct", "natural_local", "facts_supported")),
                "all four review checks must be explicitly true")
        require(isinstance(review.get("reviewer"), str) and bool(review["reviewer"].strip()), "reviewer required")
        turns = item.get("session_memory")
        require(isinstance(turns, list) and len(turns) <= 8, "session_memory must have at most eight turns")
        require(all(isinstance(t, dict) and t.get("speaker") in ("peer", "user", "other")
                    and isinstance(t.get("text"), str) and t["text"].strip() for t in turns), "invalid turns")
        context = [t["text"] for t in turns]
        message = item.get("message")
        require(isinstance(message, str) and bool(message.strip()), "message must be nonempty text")
        key = fingerprint(context, message)
        require(key not in seen and fingerprint([], message) not in messages,
                "duplicate or protected conversation/message")
        seen.add(key)
        messages.add(fingerprint([], message))
        replies = item.get("replies", {})
        require(isinstance(replies, dict) and all(isinstance(replies.get(k), str) and replies[k].strip()
                for k in ("standard", "local")), "both replies required")
        require(replies["standard"].casefold().strip() != replies["local"].casefold().strip(),
                "style pair must differ")
        for style in ("STANDARD", "LOCAL"):
            rows.append({"item_id": item_id, "context": context, "speakers": [t["speaker"] for t in turns],
                         "message": message, "style": style, "response": replies[style.lower()]})
    if not rows:
        raise ValueError(f"{path}: no synthetic examples")
    return rows


def manifest(path: Path, count: int) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "items": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    # Keep the standalone validator free of the GPU training dependencies.
    protected = []
    for name in ("train", "validation"):
        for item in read_jsonl(Path(f"data/dev/{name}.jsonl")):
            protected.append({"item_id": item["id"], "context": [t["text"] for t in item["session_memory"]],
                              "message": item["message"]["text"]})
    rows = load_synthetic(args.path, protected)
    print(json.dumps(manifest(args.path, len(rows) // 2), indent=2))


if __name__ == "__main__":
    main()
