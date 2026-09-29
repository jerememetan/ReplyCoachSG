from pathlib import Path

import pytest

from replycoach.authoring import authoring_path, load_author_items, save_author_item
from replycoach.io import save_items
from replycoach.schema import DialogueItem, Message, Replies, Turn
from replycoach.team_review import (
    assigned_review_path,
    complete_review,
    completed_review_count,
    target_role_for,
)


def make_item(item_id: str = "ITEM-A-001") -> DialogueItem:
    return DialogueItem(
        id=item_id,
        split="unassigned",
        scenario="meals_and_food",
        intent="invite_or_propose",
        session_memory=[Turn(speaker="peer", text="Free after class?")],
        message=Message(text="Want to get dinner?"),
        replies=Replies(standard="Sure, where should we meet?", local="Can, where should we meet?"),
        author="Author",
        privacy_ok=True,
        status="draft_needs_team_review",
    )


@pytest.mark.parametrize("role,target", [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")])
def test_target_role_for_rotates_to_next_role(role, target):
    assert target_role_for(role) == target


def test_target_role_for_normalizes_role_and_never_self_assigns():
    assert target_role_for(" a ") == "B"
    assert all(target_role_for(role) != role for role in "ABCDE")


@pytest.mark.parametrize("role", ["", "F", "../A"])
def test_target_role_for_rejects_invalid_roles(role):
    with pytest.raises(ValueError, match="role"):
        target_role_for(role)


def test_assigned_review_path_uses_the_target_authors_file(tmp_path: Path):
    assert assigned_review_path(" a ", tmp_path) == authoring_path("B", tmp_path)


def test_assigned_review_path_rejects_invalid_roles(tmp_path: Path):
    with pytest.raises(ValueError, match="role"):
        assigned_review_path("F", tmp_path)


def test_assigned_review_path_loads_missing_and_empty_targets_as_empty(tmp_path: Path):
    missing = assigned_review_path("A", tmp_path)
    empty = assigned_review_path("B", tmp_path)
    empty.write_text("", encoding="utf-8")

    assert load_author_items(missing) == []
    assert load_author_items(empty) == []


def test_complete_review_assigns_reviewer_and_preserves_legacy_second_fields():
    item = make_item()
    item.second_reviewer = "Legacy reviewer"
    item.second_reviewer_notes = "Legacy note"

    complete_review(item, "  Reviewer One  ")

    assert item.reviewer == "Reviewer One"
    assert item.status == "completed"
    assert item.second_reviewer == "Legacy reviewer"
    assert item.second_reviewer_notes == "Legacy note"
    item.validate()


def test_complete_review_allows_same_reviewer_to_edit_again():
    item = make_item()
    complete_review(item, "Reviewer One")
    item.reviewer_notes = "Updated after review"

    complete_review(item, " Reviewer One ")

    assert item.reviewer == "Reviewer One"
    assert item.reviewer_notes == "Updated after review"
    assert item.status == "completed"


def test_complete_review_blocks_a_different_reviewer():
    item = make_item()
    complete_review(item, "Reviewer One")

    with pytest.raises(ValueError, match="already reviewed"):
        complete_review(item, "Reviewer Two")


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_complete_review_requires_a_nonblank_reviewer_name(name):
    with pytest.raises(ValueError, match="reviewer"):
        complete_review(make_item(), name)


def test_complete_review_requires_unassigned_split():
    item = make_item()
    item.split = "dev"

    with pytest.raises(ValueError, match="unassigned"):
        complete_review(item, "Reviewer One")


def test_complete_review_rejects_an_already_split_item():
    item = make_item()
    item.split = "test"

    with pytest.raises(ValueError, match="unassigned"):
        complete_review(item, "Reviewer One")


def test_completed_review_count_counts_only_completed_items_with_reviewers():
    reviewed = make_item()
    complete_review(reviewed, "Reviewer One")
    no_reviewer = make_item("ITEM-A-002")
    no_reviewer.status = "completed"
    incomplete = make_item("ITEM-A-003")
    incomplete.reviewer = "Reviewer Two"

    assert completed_review_count([reviewed, no_reviewer, incomplete]) == 1


def test_assigned_file_can_be_loaded_and_updated_without_touching_other_fields(tmp_path: Path):
    path = assigned_review_path("A", tmp_path)
    item = make_item("ITEM-B-001")
    item.second_reviewer = "Legacy reviewer"
    item.second_reviewer_notes = "Legacy note"
    save_items(path, [item])

    loaded = load_author_items(path)
    complete_review(loaded[0], "Reviewer One")

    assert loaded[0].reviewer == "Reviewer One"
    assert loaded[0].second_reviewer == "Legacy reviewer"
    assert loaded[0].second_reviewer_notes == "Legacy note"
