"""Structured, two-pass reviewer for the team-authored pilot JSONL data."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from replycoach.io import PILOT_PATH, load_items, save_items
from replycoach.review import (
    ReviewPermissionError,
    assign_reviewer,
    format_position,
    format_turns,
    parse_turns,
)
from replycoach.schema import LANGUAGE_TAGS, Message, Replies
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


def _split_terms(text: str) -> list[str]:
    return [term.strip() for term in text.split(",") if term.strip()]


def _current_role(item, reviewer_name: str) -> tuple[str | None, str | None]:
    """Return the current reviewer role without changing the loaded item."""
    if not reviewer_name.strip():
        return None, "Enter your name before saving."
    preview = replace(item)
    try:
        return assign_reviewer(preview, reviewer_name), None
    except ReviewPermissionError as exc:
        return None, str(exc)


def _navigate(delta: int, item_count: int) -> None:
    st.session_state.review_index = max(0, min(st.session_state.review_index + delta, item_count - 1))
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Reply Coach — Pilot Reviewer", layout="wide")
    st.title("Pilot data reviewer")
    st.caption("Review one team-authored pilot item at a time. Saving writes directly to the shared JSONL file.")

    if "review_index" not in st.session_state:
        st.session_state.review_index = 0
    if "reviewer_name" not in st.session_state:
        st.session_state.reviewer_name = ""

    items = load_items(PILOT_PATH)
    if not items:
        st.error("No pilot items were found.")
        return
    st.session_state.review_index = min(st.session_state.review_index, len(items) - 1)
    item = items[st.session_state.review_index]

    reviewer_name = st.text_input("Your name", key="reviewer_name", placeholder="Enter your name before saving")
    reviewed_count = sum(bool(row.reviewer) for row in items)
    st.caption(f"{reviewed_count} of {len(items)} items have a first reviewer.")

    previous, position, next_item = st.columns([1, 2, 1])
    previous.button("← Previous", disabled=st.session_state.review_index == 0, on_click=_navigate, args=(-1, len(items)))
    position.markdown(f"### {format_position(item.id, st.session_state.review_index, len(items))}")
    next_item.button("Next →", disabled=st.session_state.review_index == len(items) - 1, on_click=_navigate, args=(1, len(items)))

    role, permission_error = _current_role(item, reviewer_name)
    st.info(f"First reviewer: {item.reviewer or '—'}  ·  Second reviewer: {item.second_reviewer or '—'}")
    if permission_error:
        st.warning(permission_error)

    with st.form(f"review-{item.id}"):
        left, right = st.columns(2)
        with left:
            scenario = st.selectbox(
                "Scenario",
                SCENARIOS,
                index=scenario_selection_index(item.scenario),
                placeholder="Choose the dominant conversation topic",
                format_func=format_scenario,
            )
            if scenario_selection_index(item.scenario) is None:
                st.caption(f"Existing label: `{item.scenario}` — choose a canonical scenario before saving.")
            intent = st.selectbox(
                "Intent",
                INTENTS,
                index=intent_selection_index(item.intent),
                placeholder="Choose the incoming message's intent",
                format_func=format_intent,
            )
            if intent is None:
                st.caption(f"Existing label: `{item.intent}` — choose a canonical intent before saving.")
            conversation = st.text_area(
                "Recent conversation",
                format_turns(item.session_memory),
                height=150,
                help="One turn per line: peer:, user:, or other: followed by the message.",
            )
            message = st.text_area("Incoming message", item.message.text, height=90)
            language_mix = st.multiselect("Language tags", LANGUAGE_TAGS, default=item.message.language_mix)
            local_terms = st.text_input("Local terms", ", ".join(item.message.local_terms))
        with right:
            standard = st.text_area("Standard reply", item.replies.standard, height=100)
            local = st.text_area("Local reply", item.replies.local, height=100)
            privacy_ok = st.checkbox("Privacy-safe", value=item.privacy_ok)
            status = st.text_input("Review status", item.status)
            st.text_area("Author / draft notes", item.notes, disabled=True, height=80)
            if role == "second":
                st.text_area("First reviewer comments", item.reviewer_notes or "(No first-reviewer comments yet.)", disabled=True, height=100)
                review_notes = st.text_area("Second reviewer comments", item.second_reviewer_notes, height=100)
            else:
                review_notes = st.text_area("First reviewer comments", item.reviewer_notes, height=100)

        save = st.form_submit_button("Save current item", type="primary", disabled=permission_error is not None)

    if not save:
        return

    try:
        item.scenario = require_scenario(scenario)
        item.intent = require_intent(intent)
        item.session_memory = parse_turns(conversation)
        item.message = Message(text=message.strip(), language_mix=list(language_mix), local_terms=_split_terms(local_terms))
        item.replies = Replies(standard=standard.strip(), local=local.strip())
        item.privacy_ok = privacy_ok
        item.status = status.strip()
        saved_role = assign_reviewer(item, reviewer_name)
        if saved_role == "first":
            item.reviewer_notes = review_notes.strip()
        else:
            item.second_reviewer_notes = review_notes.strip()
        item.validate()
        save_items(PILOT_PATH, items)
    except (ReviewPermissionError, ValueError) as exc:
        st.error(f"Not saved: {exc}")
        return

    st.success(f"Saved {item.id} as the {saved_role} reviewer.")


if __name__ == "__main__":
    main()
