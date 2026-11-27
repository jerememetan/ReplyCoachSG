"""Append-only experiment logger used by every role."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from replycoach.io import ROOT

LOG_PATH = ROOT / "experiments" / "logs" / "runs.jsonl"


def log_event(event: str, payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    out = path or LOG_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
