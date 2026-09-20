import pytest

from replycoach.taxonomy import (
    INTENTS,
    INTENT_DEFINITIONS,
    SCENARIOS,
    SCENARIO_DEFINITIONS,
    format_intent,
    format_scenario,
    intent_selection_index,
    require_intent,
    require_scenario,
    scenario_selection_index,
)


def test_taxonomy_contains_the_approved_unique_values_and_definitions():
    assert SCENARIOS == (
        "meals_and_food",
        "study_sessions_and_spaces",
        "course_materials",
        "assignments_and_group_projects",
        "classes_and_tutorials",
        "meetups_and_transport",
        "social_events_and_outings",
        "sports_and_exercise",
        "campus_services_and_equipment",
    )
    assert INTENTS == (
        "invite_or_propose",
        "request_information_or_status",
        "request_help_or_resource",
        "confirm_plan",
        "change_or_reschedule",
        "decline",
        "inform_or_update",
        "remind_or_nudge",
        "apologise_or_explain",
    )
    assert len(SCENARIOS) == len(set(SCENARIOS))
    assert len(INTENTS) == len(set(INTENTS))
    assert set(SCENARIO_DEFINITIONS) == set(SCENARIOS)
    assert set(INTENT_DEFINITIONS) == set(INTENTS)


def test_legacy_labels_require_manual_reselection():
    assert scenario_selection_index("weekend_plan") is None
    assert intent_selection_index("ask_status") is None
    with pytest.raises(ValueError, match="scenario"):
        require_scenario("weekend_plan")
    with pytest.raises(ValueError, match="intent"):
        require_intent("ask_status")


def test_canonical_labels_select_validate_and_format_with_definitions():
    assert scenario_selection_index("meals_and_food") == 0
    assert intent_selection_index("request_information_or_status") == 1
    assert require_scenario("meals_and_food") == "meals_and_food"
    assert require_intent("request_information_or_status") == "request_information_or_status"
    assert format_scenario("meals_and_food").startswith("meals_and_food — ")
    assert format_intent("request_information_or_status").startswith("request_information_or_status — ")
