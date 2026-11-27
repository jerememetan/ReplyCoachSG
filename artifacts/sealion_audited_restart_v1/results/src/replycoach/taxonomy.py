"""Canonical scenario and incoming-message intent labels."""

from __future__ import annotations


SCENARIOS = (
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

SCENARIO_DEFINITIONS = {
    "meals_and_food": "Choosing, obtaining, or meeting for food or drinks.",
    "study_sessions_and_spaces": "Arranging peer study or finding and sharing study space.",
    "course_materials": "Requesting, locating, or sharing notes, slides, readings, or recordings.",
    "assignments_and_group_projects": "Coordinating assessed work, contributions, deadlines, or presentations.",
    "classes_and_tutorials": "Coordinating attendance, time, room, or changes to scheduled teaching.",
    "meetups_and_transport": "Travelling, arriving, waiting, or locating one another for a meetup.",
    "social_events_and_outings": "Coordinating non-food, non-sport leisure activities, clubs, or outings.",
    "sports_and_exercise": "Coordinating participation in sport, exercise, or physical recreation.",
    "campus_services_and_equipment": "Using printers, queues, chargers, devices, or shared equipment.",
}

INTENTS = (
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

INTENT_DEFINITIONS = {
    "invite_or_propose": "Suggests a new activity, plan, or option.",
    "request_information_or_status": "Asks for a fact, location, progress, readiness, or current state.",
    "request_help_or_resource": "Asks the recipient to perform a new action or provide something.",
    "confirm_plan": "Confirms or checks an arrangement that has already been discussed.",
    "change_or_reschedule": "Asks to modify an existing arrangement.",
    "decline": "Refuses or says they cannot participate in a request, invitation, or plan.",
    "inform_or_update": "Provides new information without primarily requesting action or apologising.",
    "remind_or_nudge": "Prompts completion of an obligation already established in the conversation.",
    "apologise_or_explain": "Apologises for or explains a problem, absence, failure, or delay.",
}


def _selection_index(value: str, choices: tuple[str, ...]) -> int | None:
    return choices.index(value) if value in choices else None


def scenario_selection_index(scenario: str) -> int | None:
    return _selection_index(scenario, SCENARIOS)


def intent_selection_index(intent: str) -> int | None:
    return _selection_index(intent, INTENTS)


def require_scenario(scenario: str | None) -> str:
    if scenario not in SCENARIOS:
        raise ValueError("Choose a scenario from the approved list.")
    return scenario


def require_intent(intent: str | None) -> str:
    if intent not in INTENTS:
        raise ValueError("Choose an intent from the approved list.")
    return intent


def format_scenario(scenario: str) -> str:
    return f"{scenario} — {SCENARIO_DEFINITIONS[scenario]}"


def format_intent(intent: str) -> str:
    return f"{intent} — {INTENT_DEFINITIONS[intent]}"
