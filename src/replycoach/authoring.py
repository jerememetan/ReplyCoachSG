"""Pure helpers for team-authored dataset records."""

from __future__ import annotations

import re
from pathlib import Path

from replycoach.io import DATA_DIR, load_items, save_items
from replycoach.schema import DialogueItem

AUTHOR_ROLES = ("A", "B", "C", "D", "E")
AUTHORING_TARGET = 60
AUTHORING_SCENARIOS = (
    "meal",
    "running_late",
    "library_or_seats",
    "notes_or_slides",
    "cafe_study",
    "weekend_plan",
    "mrt_or_gate_meetup",
    "deadline_tonight",
    "polite_decline",
    "confirm_time_or_place",
    "room_change",
    "printing_or_queue",
    "rain_reschedule",
    "sports_or_walk",
    "group_chat_nudge",
    "arrival",
    "wait_a_moment",
    "split_groups",
    "study_environment",
    "after_class_food",
)


def normalise_role(role: str) -> str:
    cleaned = role.strip().upper()
    if cleaned not in AUTHOR_ROLES:
        raise ValueError(f"role must be one of {AUTHOR_ROLES}")
    return cleaned


def authoring_path(role: str, directory: Path | None = None) -> Path:
    """Return the role's authoring file inside the approved directory."""
    role = normalise_role(role)
    return (directory or DATA_DIR / "authoring") / f"items_{role}.jsonl"


def load_author_items(path: Path) -> list[DialogueItem]:
    """Load an authoring file, treating a missing file as an empty collection."""
    if not path.exists():
        return []
    items = load_items(path)
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate item IDs")
    return items


def next_item_id(role: str, items: list[DialogueItem]) -> str:
    """Allocate the next stable role-specific item identifier."""
    role = normalise_role(role)
    pattern = re.compile(rf"^ITEM-{role}-(\d+)$")
    used = [int(match.group(1)) for item in items if (match := pattern.match(item.id))]
    return f"ITEM-{role}-{max(used, default=0) + 1:03d}"


def save_author_item(path: Path, item: DialogueItem) -> None:
    """Create or replace one unassigned author record using an atomic write."""
    if item.split != "unassigned":
        raise ValueError("authoring items must remain unassigned")
    if item.reviewer or item.second_reviewer:
        raise ValueError("authors cannot set reviewer fields")
    item.validate()
    items = load_author_items(path)
    for index, existing in enumerate(items):
        if existing.id == item.id:
            items[index] = item
            break
    else:
        items.append(item)
    save_items(path, items)
