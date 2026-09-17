"""Shared data contracts used by every role.

Friday Week 4 goal: every member can load the same data format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

STYLES = ("STANDARD", "LOCAL")
SOURCES = ("V0", "V1", "V2", "V3", "V4", "V5", "V6", "fallback", "gold")
SPLITS = ("pilot", "dev", "test")
SPEAKERS = ("peer", "user", "other")
LANGUAGE_TAGS = ("en", "singlish", "zh", "pinyin")


@dataclass
class Turn:
    speaker: str
    text: str

    def validate(self) -> None:
        if self.speaker not in SPEAKERS:
            raise ValueError(f"speaker must be one of {SPEAKERS}, got {self.speaker!r}")
        if not str(self.text).strip():
            raise ValueError("turn text must be non-empty")


@dataclass
class Message:
    text: str
    language_mix: list[str] = field(default_factory=lambda: ["en"])
    local_terms: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not str(self.text).strip():
            raise ValueError("message.text must be non-empty")
        unknown = [tag for tag in self.language_mix if tag not in LANGUAGE_TAGS]
        if unknown:
            raise ValueError(f"unknown language tags: {unknown}")


@dataclass
class Replies:
    standard: str
    local: str

    def validate(self) -> None:
        if not str(self.standard).strip():
            raise ValueError("standard reply must be non-empty")
        if not str(self.local).strip():
            raise ValueError("local reply must be non-empty")


@dataclass
class DialogueItem:
    """One team-authored or public-seed example."""

    id: str
    split: str
    scenario: str
    intent: str
    session_memory: list[Turn]
    message: Message
    replies: Replies
    author: str = "unassigned"
    reviewer: str = ""
    second_reviewer: str = ""
    reviewer_notes: str = ""
    second_reviewer_notes: str = ""
    notes: str = ""
    privacy_ok: bool = True
    licence: str = "team-authored"
    origin: str = "team"
    status: str = "draft_needs_team_review"

    def validate(self) -> None:
        if not self.id:
            raise ValueError("id is required")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if not self.privacy_ok:
            raise ValueError(f"{self.id} failed privacy_ok")
        for turn in self.session_memory:
            turn.validate()
        self.message.validate()
        self.replies.validate()

    def memory_texts(self) -> list[str]:
        return [turn.text for turn in self.session_memory]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DialogueItem":
        memory = [Turn(**turn) if not isinstance(turn, Turn) else turn for turn in raw.get("session_memory", [])]
        message_raw = raw.get("message", {})
        replies_raw = raw.get("replies", {})
        if isinstance(message_raw, str):
            message = Message(text=message_raw)
        else:
            message = Message(**message_raw)
        replies = Replies(**replies_raw) if not isinstance(replies_raw, Replies) else replies_raw
        item = cls(
            id=raw["id"],
            split=raw.get("split", "pilot"),
            scenario=raw.get("scenario", ""),
            intent=raw.get("intent", ""),
            session_memory=memory,
            message=message,
            replies=replies,
            author=raw.get("author", "unassigned"),
            reviewer=raw.get("reviewer", ""),
            second_reviewer=raw.get("second_reviewer", ""),
            reviewer_notes=raw.get("reviewer_notes", ""),
            second_reviewer_notes=raw.get("second_reviewer_notes", ""),
            notes=raw.get("notes", ""),
            privacy_ok=bool(raw.get("privacy_ok", True)),
            licence=raw.get("licence", "team-authored"),
            origin=raw.get("origin", "team"),
            status=raw.get("status", "draft_needs_team_review"),
        )
        item.validate()
        return item


@dataclass
class Candidate:
    """Shared candidate format for retrieval, generation, and reranking."""

    context: list[str]
    message: str
    candidate_text: str
    style: Literal["STANDARD", "LOCAL"]
    source: str
    score: float
    item_id: str | None = None
    normalised_message: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.style not in STYLES:
            raise ValueError(f"style must be STANDARD or LOCAL, got {self.style!r}")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if not str(self.candidate_text).strip():
            raise ValueError("candidate_text must be non-empty")
        if not isinstance(self.score, (int, float)):
            raise ValueError("score must be numeric")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = float(self.score)
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Candidate":
        candidate = cls(
            context=list(raw.get("context") or []),
            message=raw["message"],
            candidate_text=raw["candidate_text"],
            style=raw["style"],
            source=raw["source"],
            score=float(raw["score"]),
            item_id=raw.get("item_id"),
            normalised_message=raw.get("normalised_message"),
            meta=dict(raw.get("meta") or {}),
        )
        candidate.validate()
        return candidate


def require_items(items: Iterable[DialogueItem]) -> list[DialogueItem]:
    loaded = list(items)
    if not loaded:
        raise ValueError("expected at least one dialogue item")
    for item in loaded:
        item.validate()
    return loaded
