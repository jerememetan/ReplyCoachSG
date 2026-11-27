"""Tokenisation that keeps Latin words and Chinese characters as retrieval units."""

from __future__ import annotations

import re
import unicodedata

LATIN_OR_CJK = re.compile(
    r"[\u4e00-\u9fff]|[a-z0-9]+(?:'[a-z]+)?",
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric words plus individual CJK characters."""
    cleaned = normalize_unicode(text).lower()
    return LATIN_OR_CJK.findall(cleaned)


def tokenize_join(text: str) -> str:
    return " ".join(tokenize(text))
