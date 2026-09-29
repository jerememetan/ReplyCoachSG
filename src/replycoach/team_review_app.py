"""Streamlit reviewer for assigned team-authored items."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import streamlit as st

from replycoach.authoring import AUTHOR_ROLES, load_author_items
from replycoach.io import save_items
from replycoach.review import format_position, format_turns, parse_turns
from replycoach.schema import LANGUAGE_TAGS, DialogueItem, Message, Replies
from replycoach.taxonomy import (
    INTENTS,
    SCENARIOS,
    format_intent,
    format_scenario,
    intent_selection_index,
    require_intent,
    require_scenario,
    scenario_selection_index,
)
from replycoach.team_review import assigned_review_path, complete_review, completed_review_count, target_role_for


def _split_terms(text: str) -> list[str]:
    return [term.strip() for term in text.split(",") if term.strip()]


def load_review_items(path: Path) -> tuple[list[DialogueItem], str | None]:
    """Read an assigned file, reporting malformed rows without changing it."""
    try:
        return load_author_items(path), None
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return [], f"Could not read assigned file {path.name}: {exc}"


def save_review(
    path: Path,
    items: list[DialogueItem],
    index: int,
    reviewer_name: str,
    *,
    scenario: str | None,
    intent: str | None,
    conversation: str,
    message: str,
    language_mix: list[str],
    local_terms: str,
    standard: str,
    local: str,
    privacy_ok: bool,
    reviewer_notes: str,
) -> DialogueItem:
    """Validate, complete, and atomically persist one assigned review."""
    item = replace(items[index])
    item.scenario = require_scenario(scenario)
    item.intent = require_intent(intent)
    item.session_memory = parse_turns(conversation)
    item.message = Message(
        text=message.strip(),
        language_mix=list(language_mix),
        local_terms=_split_terms(local_terms),
    )
    item.replies = Replies(standard=standard.strip(), local=local.strip())
    item.privacy_ok = privacy_ok
    item.reviewer_notes = reviewer_notes.strip()
    complete_review(item, reviewer_name)
    updated = list(items)
    updated[index] = item
    save_items(path, updated)
    return item


def _navigate(delta: int, count: int) -> None:
    st.session_state.team_review_index = max(0, min(st.session_state.team_review_index + delta, count - 1))


def main() -> None:
    st.set_page_config(page_title="Reply Coach — Team Reviewer", layout="wide")
    st.title("Team data reviewer")
    st.caption("Review your assigned teammate's examples. Saving writes to that teammate's authoring file.")

    identity, role_column = st.columns([3, 1])
    with identity:
        reviewer_name = st.text_input("Your name", key="team_reviewer_name", placeholder="Enter your name before saving")
    with role_column:
        role = st.selectbox("Your role", AUTHOR_ROLES, key="team_reviewer_role")

    if st.session_state.get("team_review_selected_role") != role:
        st.session_state.team_review_index = 0
        st.session_state.team_review_selected_role = role

    target_role = target_role_for(role)
    path = assigned_review_path(role)
    st.info(f"Role {role} reviews Role {target_role} · data/authoring/{path.name}")

    items, load_error = load_review_items(path)
    if load_error:
        st.error(load_error)
        return
    if not items:
        st.warning(f"Assigned file {path.name} is missing or empty. Ask Role {target_role} to save their drafts first.")
        return

    st.session_state.team_review_index = min(st.session_state.team_review_index, len(items) - 1)
    index = st.session_state.team_review_index
    item = items[index]
    completed = completed_review_count(items)
    st.progress(completed / len(items), text=f"{completed} of {len(items)} reviews completed")

    previous, position, next_item = st.columns([1, 2, 1])
    previous.button("← Previous", disabled=index == 0, on_click=_navigate, args=(-1, len(items)))
    position.markdown(f"### {format_position(item.id, index, len(items))}")
    next_item.button("Next →", disabled=index == len(items) - 1, on_click=_navigate, args=(1, len(items)))

    st.info(f"Status: {item.status} · Reviewer: {item.reviewer or 'Unassigned'}")
    reviewer = reviewer_name.strip()
    permission_error = None
    if not reviewer:
        permission_error = "Enter your name before saving."
    elif item.reviewer and item.reviewer != reviewer:
        permission_error = f"This item was already reviewed by {item.reviewer}."
    if permission_error:
        st.warning(permission_error)

    with st.form(f"team-review-{role}-{item.id}"):
        left, right = st.columns(2)
        with left:
            scenario_index = scenario_selection_index(item.scenario)
            scenario = st.selectbox(
                "Scenario", SCENARIOS, index=scenario_index,
                placeholder="Choose a canonical scenario", format_func=format_scenario,
            )
            if scenario_index is None:
                st.caption(f"Existing label: `{item.scenario}` — choose a canonical scenario before saving.")
            intent_index = intent_selection_index(item.intent)
            intent = st.selectbox(
                "Incoming message intent", INTENTS, index=intent_index,
                placeholder="Choose the incoming message's intent", format_func=format_intent,
            )
            if intent_index is None:
                st.caption(f"Existing label: `{item.intent}` — choose a canonical intent before saving.")
            conversation = st.text_area(
                "Recent conversation", format_turns(item.session_memory), height=150,
                help="One turn per line: peer:, user:, or other: followed by the message.",
            )
            message = st.text_area("Incoming peer message", item.message.text, height=100)
            language_mix = st.multiselect("Incoming message language tags", LANGUAGE_TAGS, default=item.message.language_mix)
            local_terms = st.text_input("Local terms in the incoming message", ", ".join(item.message.local_terms))
        with right:
            standard = st.text_area("Standard reply", item.replies.standard, height=120)
            local = st.text_area("Light local reply", item.replies.local, height=120)
            privacy_ok = st.checkbox(
                "I confirm this item contains no identifying information.", value=item.privacy_ok,
            )
            st.text_area("Author notes", item.notes, height=100, disabled=True)
            reviewer_notes = st.text_area("Reviewer comments", item.reviewer_notes, height=100)
            if item.second_reviewer or item.second_reviewer_notes:
                st.caption(f"Legacy second reviewer: {item.second_reviewer or '—'}")
                st.text_area("Legacy second reviewer comments", item.second_reviewer_notes, height=80, disabled=True)

        save = st.form_submit_button("Save review", type="primary", disabled=permission_error is not None)

    if not save:
        return
    try:
        saved = save_review(
            path, items, index, reviewer_name,
            scenario=scenario, intent=intent, conversation=conversation,
            message=message, language_mix=list(language_mix), local_terms=local_terms,
            standard=standard, local=local, privacy_ok=privacy_ok,
            reviewer_notes=reviewer_notes,
        )
    except (OSError, ValueError) as exc:
        st.error(f"Not saved: {exc}")
        return
    st.success(f"Saved review for {saved.id}.")


if __name__ == "__main__":
    main()
