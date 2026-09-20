from pathlib import Path

import pytest

from replycoach.authoring import AUTHORING_TARGET, authoring_path, load_author_items, next_item_id, save_author_item
from replycoach.schema import DialogueItem, Message, Replies, Turn


def make_item(item_id: str = "ITEM-E-001", standard: str = "Sure, where should we meet?") -> DialogueItem:
    return DialogueItem(
        id=item_id,
        split="unassigned",
        scenario="meal",
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
