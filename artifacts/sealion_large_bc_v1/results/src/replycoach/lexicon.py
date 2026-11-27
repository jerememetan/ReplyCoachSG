"""Local-term lexicon loader for V1 detection and normalisation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from replycoach.io import LEXICON_PATH
from replycoach.tokenize import tokenize


@dataclass(frozen=True)
class LocalTerm:
    surface: str
    canonical: str
    kind: str
    notes: str = ""
    apply: bool = True


def load_lexicon(path: Path | None = None) -> list[LocalTerm]:
    raw = json.loads((path or LEXICON_PATH).read_text(encoding="utf-8"))
    terms = []
    for row in raw["terms"]:
        terms.append(
            LocalTerm(
                surface=row["surface"],
                canonical=row["canonical"],
                kind=row.get("kind", "other"),
                notes=row.get("notes", ""),
                apply=bool(row.get("apply", True)),
            )
        )
    return terms


def longest_first(terms: list[LocalTerm]) -> list[LocalTerm]:
    return sorted(terms, key=lambda term: len(tokenize(term.surface)), reverse=True)
