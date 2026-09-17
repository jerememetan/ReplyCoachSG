import pytest

from replycoach.review import ReviewPermissionError, assign_reviewer, format_position, parse_turns
from replycoach.schema import DialogueItem, Message, Replies


def make_item() -> DialogueItem:
    return DialogueItem(
        id="PILOT-X",
        split="pilot",
        scenario="lunch",
        intent="propose_meal",
        session_memory=[],
        message=Message(text="Want to eat?"),
        replies=Replies(standard="Want to have lunch?", local="Want to makan?"),
    )


def test_assign_reviewer_uses_first_empty_slot():
    item = make_item()
    assert assign_reviewer(item, "Ava") == "first"
    assert item.reviewer == "Ava"
    assert assign_reviewer(item, "Ben") == "second"
    assert item.second_reviewer == "Ben"


def test_assign_reviewer_keeps_existing_role_and_blocks_a_third_person():
    item = make_item()
    item.reviewer, item.second_reviewer = "Ava", "Ben"
    assert assign_reviewer(item, "Ava") == "first"
    assert assign_reviewer(item, "Ben") == "second"
    with pytest.raises(ReviewPermissionError, match="two reviewers"):
        assign_reviewer(item, "Chen")


def test_parse_turns_accepts_known_speakers_and_rejects_unknown_ones():
    turns = parse_turns("peer: lecture ending soon\nuser: I am hungry")
    assert [(turn.speaker, turn.text) for turn in turns] == [
        ("peer", "lecture ending soon"),
        ("user", "I am hungry"),
    ]
    with pytest.raises(ValueError, match="peer, user, or other"):
        parse_turns("friend: see you")


def test_format_position_uses_a_real_line_break_without_extra_symbols():
    assert format_position("PILOT-001", 0, 20) == "PILOT-001\n1 of 20"
