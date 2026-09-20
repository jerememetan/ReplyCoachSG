import pytest

from replycoach import review
from replycoach.review import ReviewPermissionError, assign_reviewer, format_position, parse_turns
from replycoach.schema import DialogueItem, Message, Replies
from replycoach.taxonomy import INTENTS, SCENARIOS


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


def test_review_uses_the_shared_taxonomy():
    assert review.PILOT_SCENARIOS is SCENARIOS
    assert review.PILOT_INTENTS is INTENTS


def test_legacy_pilot_scenario_requires_manual_reselection():
    assert review.scenario_selection_index("lunch_after_lecture") is None


def test_pilot_intents_are_unique_and_describe_incoming_messages():
    assert len(review.PILOT_INTENTS) == len(set(review.PILOT_INTENTS))
    assert "invite_or_propose" in review.PILOT_INTENTS
    assert "request_information_or_status" in review.PILOT_INTENTS
    assert "ask_status" not in review.PILOT_INTENTS


def test_intent_selection_index_selects_canonical_values_only():
    assert review.intent_selection_index("request_information_or_status") == review.PILOT_INTENTS.index(
        "request_information_or_status"
    )
    assert review.intent_selection_index("ask_status") is None
    assert review.intent_selection_index("apologise_and_eta") is None


def test_scenario_selection_index_selects_canonical_values_only():
    assert review.scenario_selection_index("meals_and_food") == review.PILOT_SCENARIOS.index("meals_and_food")
    assert review.scenario_selection_index("lunch_after_lecture") is None


def test_require_intent_rejects_an_unselected_or_unknown_value():
    assert review.require_intent("confirm_plan") == "confirm_plan"
    with pytest.raises(ValueError, match="Choose an intent"):
        review.require_intent(None)
    with pytest.raises(ValueError, match="Choose an intent"):
        review.require_intent("custom_intent")


def test_require_scenario_rejects_an_unselected_or_unknown_value():
    assert review.require_scenario("meals_and_food") == "meals_and_food"
    with pytest.raises(ValueError, match="Choose a scenario"):
        review.require_scenario(None)
    with pytest.raises(ValueError, match="Choose a scenario"):
        review.require_scenario("lunch_after_lecture")
