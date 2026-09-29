"""Helpers for the single-review team workflow."""

from __future__ import annotations

from pathlib import Path

from replycoach.authoring import authoring_path, normalise_role
from replycoach.schema import DialogueItem

_REVIEW_TARGETS = {"A": "B", "B": "C", "C": "D", "D": "E", "E": "A"}


def target_role_for(role: str) -> str:
    """Return the next role in the fixed review cycle."""
    role = normalise_role(role)
    return _REVIEW_TARGETS[role]


def assigned_review_path(role: str, directory: Path | None = None) -> Path:
    """Return the authoring file assigned to this role's reviewer."""
    return authoring_path(target_role_for(role), directory)


def complete_review(item: DialogueItem, name: str) -> None:
    """Record one completed review, allowing its reviewer to edit it again."""
    reviewer = name.strip()
    if not reviewer:
        raise ValueError("reviewer name is required")
    if item.split != "unassigned":
        raise ValueError("review items must remain unassigned until review is complete")
    if item.reviewer and item.reviewer != reviewer:
        raise ValueError(f"item is already reviewed by {item.reviewer}")

    item.reviewer = reviewer
    item.status = "completed"
    item.validate()


def completed_review_count(items: list[DialogueItem]) -> int:
    """Count records with both a reviewer and completed status."""
    return sum(bool(item.reviewer.strip()) and item.status == "completed" for item in items)
