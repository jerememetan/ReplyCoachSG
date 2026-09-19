"""Pure helpers for the pilot-data reviewer application."""

from __future__ import annotations

from replycoach.schema import DialogueItem, SPEAKERS, Turn


PILOT_SCENARIOS = (
    "lunch_after_lecture",
    "late_to_meeting",
    "library_seats",
    "share_notes",
    "cafe_study",
    "weekend_makan",
    "mrt_meetup",
    "deadline_tonight",
    "cannot_make_it",
    "confirm_time",
    "dinner_near_campus",
    "wait_at_gate",
    "group_chat_nudge",
    "tutorial_room",
    "print_before_class",
    "rec_sports",
    "rain_change_plans",
    "supper_jio",
    "assignment_peer_check",
    "arrived_waiting",
)


class ReviewPermissionError(ValueError):
    """Raised when a person cannot occupy a review slot."""


def assign_reviewer(item: DialogueItem, name: str) -> str:
    """Assign a name to its existing review pass, or the next free pass."""
    reviewer = name.strip()
    if not reviewer:
        raise ReviewPermissionError("Enter your name before saving.")
    if not item.reviewer or item.reviewer == reviewer:
        item.reviewer = reviewer
        return "first"
    if not item.second_reviewer or item.second_reviewer == reviewer:
        item.second_reviewer = reviewer
        return "second"
    raise ReviewPermissionError("This item already has two reviewers.")


def parse_turns(text: str) -> list[Turn]:
    """Parse one `peer: text` / `user: text` / `other: text` turn per line."""
    turns: list[Turn] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        cleaned = line.strip()
        if not cleaned:
            continue
        if ":" not in cleaned:
            raise ValueError(f"Line {line_no} must start with peer:, user:, or other:.")
        speaker, message = cleaned.split(":", maxsplit=1)
        speaker, message = speaker.strip().lower(), message.strip()
        if speaker not in SPEAKERS:
            raise ValueError(f"Line {line_no} speaker must be peer, user, or other.")
        if not message:
            raise ValueError(f"Line {line_no} must contain message text.")
        turns.append(Turn(speaker=speaker, text=message))
    return turns


def format_turns(turns: list[Turn]) -> str:
    """Format stored turns into the editable reviewer-app convention."""
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)


def format_position(item_id: str, index: int, total: int) -> str:
    """Format an item identifier and its one-based review position."""
    return f"{item_id}\n{index + 1} of {total}"
