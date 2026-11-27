"""Read-only, explicitly identified reply banks for retrieval demos."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from replycoach.io import PILOT_PATH, read_jsonl
from replycoach.schema import DialogueItem


@dataclass
class ReplyBank:
    items: list[DialogueItem]
    path: Path
    label: str
    serialized: str
    fingerprint: str


def validate_development_row(row: dict) -> None:
    if row.get("split") != "dev":
        raise ValueError("split must be dev; test/unassigned records are excluded")
    if row.get("status") != "completed":
        raise ValueError("status must be completed")
    author = row.get("author", "")
    reviewer = row.get("reviewer", "")
    if not isinstance(author, str) or not author.strip() or author.strip().casefold() == "unassigned":
        raise ValueError("a named author is required")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("an independent reviewer is required")
    if author.strip().casefold() == reviewer.strip().casefold():
        raise ValueError("reviewer must be independent of the author")
    if row.get("privacy_ok") is not True:
        raise ValueError("privacy_ok must explicitly be true")


def load_reply_bank(path: Path | None = None) -> ReplyBank:
    """Default to pilot drafts; explicit files must contain only reviewed dev items."""
    is_pilot = path is None
    selected_path = PILOT_PATH if is_pilot else Path(path)
    rows = read_jsonl(selected_path)
    if not rows:
        raise ValueError("reply bank is empty")
    items = []
    seen = set()
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"record {position}: expected a JSON object")
        record_id = row.get("id", f"record {position}")
        try:
            if not is_pilot:
                validate_development_row(row)
            item = DialogueItem.from_dict(row)
            if not isinstance(item.id, str) or not item.id.strip():
                raise ValueError("id must be a non-empty string")
            if item.id in seen:
                raise ValueError(f"duplicate id {item.id}")
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise ValueError(f"{record_id}: {exc}") from exc
        seen.add(item.id)
        items.append(item)
    serialized = json.dumps([item.to_dict() for item in items], ensure_ascii=False, sort_keys=True)
    return ReplyBank(
        items=items,
        path=selected_path.resolve(),
        label="Pilot examples (drafts)" if is_pilot else "Reviewed development examples",
        serialized=serialized,
        fingerprint=sha256(serialized.encode("utf-8")).hexdigest(),
    )
