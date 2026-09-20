"""Pure helpers for team-authored dataset records."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from replycoach.io import DATA_DIR, load_items, save_items
from replycoach.schema import DialogueItem
from replycoach.taxonomy import INTENTS, SCENARIOS

AUTHOR_ROLES = ("A", "B", "C", "D", "E")
AUTHORING_TARGET = 60
AUTHORING_SCENARIOS = SCENARIOS


@dataclass(frozen=True)
class CombinationSuggestion:
    scenario: str
    intent: str
    count: int


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


def load_team_author_items(directory: Path | None = None) -> tuple[list[DialogueItem], list[str]]:
    """Load every valid role file and report unreadable files without changing them."""
    items: list[DialogueItem] = []
    warnings: list[str] = []
    for role in AUTHOR_ROLES:
        path = authoring_path(role, directory)
        try:
            items.extend(load_author_items(path))
        except ValueError as exc:
            warnings.append(f"Could not include {path.name}: {exc}")
    return items, warnings


def suggest_authoring_combination(
    items: list[DialogueItem],
    role: str,
    item_id: str,
) -> CombinationSuggestion:
    """Choose a deterministic least-used scenario and intent pair."""
    role = normalise_role(role)
    match = re.fullmatch(rf"ITEM-{role}-(\d+)", item_id)
    if not match:
        raise ValueError(f"item ID must match ITEM-{role}-###")
    combinations = list(product(AUTHORING_SCENARIOS, INTENTS))
    counts = Counter((item.scenario, item.intent) for item in items)
    minimum = min(counts[combination] for combination in combinations)
    least_used = [combination for combination in combinations if counts[combination] == minimum]

    item_number = int(match.group(1))
    role_index = AUTHOR_ROLES.index(role)
    scenario_index = (item_number - 1) % len(AUTHORING_SCENARIOS)
    cycle = (item_number - 1) // len(AUTHORING_SCENARIOS)
    intent_index = (cycle * len(AUTHOR_ROLES) + role_index + scenario_index) % len(INTENTS)
    preferred = (AUTHORING_SCENARIOS[scenario_index], INTENTS[intent_index])
    selected = preferred if preferred in least_used else least_used[combinations.index(preferred) % len(least_used)]
    return CombinationSuggestion(*selected, count=counts[selected])


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
