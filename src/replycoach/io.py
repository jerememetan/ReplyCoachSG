"""JSON/JSONL loaders for the shared schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from replycoach.schema import Candidate, DialogueItem

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PILOT_PATH = DATA_DIR / "pilot" / "pilot_20.jsonl"
LEXICON_PATH = DATA_DIR / "lexicons" / "local_terms.json"


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_items(path: Path | None = None) -> list[DialogueItem]:
    path = path or PILOT_PATH
    return [DialogueItem.from_dict(row) for row in read_jsonl(path)]


def save_items(path: Path, items: Iterable[DialogueItem]) -> None:
    write_jsonl(path, [item.to_dict() for item in items])


def load_candidates(path: Path) -> list[Candidate]:
    return [Candidate.from_dict(row) for row in read_jsonl(path)]


def save_candidates(path: Path, candidates: Iterable[Candidate]) -> None:
    write_jsonl(path, [candidate.to_dict() for candidate in candidates])
