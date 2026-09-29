from pathlib import Path

import pytest

from replycoach.authoring import (
    AUTHORING_SCENARIOS,
    AUTHORING_TARGET,
    authoring_path,
    load_author_items,
    load_team_author_items,
    next_item_id,
    save_author_item,
    suggest_authoring_combination,
)
from replycoach.io import save_items
from replycoach.schema import DialogueItem, Message, Replies, Turn
from replycoach.taxonomy import INTENTS, SCENARIOS


def make_item(item_id: str = "ITEM-E-001", standard: str = "Sure, where should we meet?") -> DialogueItem:
    return DialogueItem(
        id=item_id,
        split="unassigned",
        scenario="meals_and_food",
        intent="invite_or_propose",
        session_memory=[Turn(speaker="peer", text="Free after class?")],
        message=Message(text="Want to get dinner?"),
        replies=Replies(standard=standard, local="Can, where should we meet?"),
        author="Jereme",
        privacy_ok=True,
        status="draft_needs_team_review",
    )


def test_each_author_targets_sixty_unassigned_items():
    assert AUTHORING_TARGET == 60


def test_authoring_uses_the_shared_scenarios():
    assert AUTHORING_SCENARIOS is SCENARIOS


def test_combination_suggestion_is_zero_count_deterministic_and_role_aware():
    a = suggest_authoring_combination([], "A", "ITEM-A-001")
    b = suggest_authoring_combination([], "B", "ITEM-B-001")

    assert a.count == 0
    assert a == suggest_authoring_combination([], "A", "ITEM-A-001")
    assert (a.scenario, a.intent) != (b.scenario, b.intent)


def test_combination_suggestion_prefers_an_unused_pair():
    used = make_item()

    suggestion = suggest_authoring_combination([used], "E", "ITEM-E-002")

    assert suggestion.count == 0
    assert (suggestion.scenario, suggestion.intent) != (used.scenario, used.intent)


def test_combination_suggestion_uses_the_least_represented_pairs():
    items = []
    for scenario in AUTHORING_SCENARIOS:
        for intent in INTENTS:
            item = make_item(f"ITEM-E-{len(items) + 1:03d}")
            item.scenario = scenario
            item.intent = intent
            items.append(item)
    duplicate = make_item("ITEM-E-999")
    items.append(duplicate)

    suggestion = suggest_authoring_combination(items, "E", "ITEM-E-091")

    assert suggestion.count == 1
    assert (suggestion.scenario, suggestion.intent) != (duplicate.scenario, duplicate.intent)


def test_authoring_path_accepts_only_known_roles(tmp_path: Path):
    assert authoring_path("e", tmp_path) == tmp_path / "items_E.jsonl"
    with pytest.raises(ValueError, match="role"):
        authoring_path("../outside", tmp_path)


def test_next_item_id_uses_the_highest_existing_role_number():
    items = [make_item("ITEM-E-001"), make_item("ITEM-E-003")]

    assert next_item_id("E", items) == "ITEM-E-004"
    assert next_item_id("A", []) == "ITEM-A-001"


def test_save_author_item_creates_and_updates_atomically(tmp_path: Path):
    path = authoring_path("E", tmp_path)
    save_author_item(path, make_item())
    save_author_item(path, make_item(standard="Yes, I can join you."))

    loaded = load_author_items(path)
    assert len(loaded) == 1
    assert loaded[0].replies.standard == "Yes, I can join you."
    assert list(tmp_path.glob(".items_E-*.jsonl")) == []


def test_save_author_item_rejects_replacing_a_completed_item(tmp_path: Path):
    path = authoring_path("E", tmp_path)
    reviewed = make_item()
    reviewed.status = "completed"
    reviewed.reviewer = "Reviewer One"
    save_items(path, [reviewed])

    with pytest.raises(ValueError, match="reviewed"):
        save_author_item(path, make_item(standard="Replacement draft"))

    loaded = load_author_items(path)
    assert loaded[0].status == "completed"
    assert loaded[0].reviewer == "Reviewer One"
    assert loaded[0].replies.standard == reviewed.replies.standard


def test_load_team_author_items_combines_valid_files_and_warns_for_invalid(tmp_path: Path):
    save_author_item(authoring_path("A", tmp_path), make_item("ITEM-A-001"))
    save_author_item(authoring_path("E", tmp_path), make_item("ITEM-E-001"))
    (tmp_path / "items_B.jsonl").write_text("not json\n", encoding="utf-8")

    items, warnings = load_team_author_items(tmp_path)

    assert {item.id for item in items} == {"ITEM-A-001", "ITEM-E-001"}
    assert len(warnings) == 1
    assert "items_B.jsonl" in warnings[0]
