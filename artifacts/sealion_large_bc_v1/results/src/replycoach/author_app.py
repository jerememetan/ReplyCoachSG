"""Streamlit app for writing team-authored Reply Coach dataset items."""

from __future__ import annotations

import streamlit as st

from replycoach.authoring import (
    AUTHOR_ROLES,
    AUTHORING_TARGET,
    authoring_path,
    load_author_items,
    load_team_author_items,
    next_item_id,
    save_author_item,
    suggest_authoring_combination,
)
from replycoach.review import format_turns, parse_turns
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


def _split_terms(text: str) -> list[str]:
    return [term.strip() for term in text.split(",") if term.strip()]


def main() -> None:
    st.set_page_config(page_title="Reply Coach — Data Author", layout="wide")
    st.title("Team data author")
    st.caption("Write and edit your own Reply Coach examples without touching JSONL directly.")
    st.info(
        "Write every conversation and reply yourself. Do not paste LLM-generated wording, real private chats, "
        "or identifiable personal information."
    )

    identity, role_column = st.columns([3, 1])
    with identity:
        author_name = st.text_input("Your name", placeholder="Used as the item's author")
    with role_column:
        role = st.selectbox("Your role", AUTHOR_ROLES)

    path = authoring_path(role)
    try:
        items = load_author_items(path)
    except ValueError as exc:
        st.error(f"Could not load {path.name}: {exc}")
        return

    completed = len(items)
    st.progress(
        min(completed / AUTHORING_TARGET, 1.0),
        text=f"{completed} of {AUTHORING_TARGET} items saved",
    )
    st.caption(f"Saving to `data/authoring/{path.name}` · Split assignment happens after review.")

    choices = ["Create a new item", *[item.id for item in items]]
    selected = st.selectbox("Item", choices, key=f"author_item_{role}")
    existing = next((item for item in items if item.id == selected), None)
    item_id = existing.id if existing else next_item_id(role, items)
    st.subheader(item_id)
    st.caption("New items are unassigned drafts. Role A will assign development and test splits later.")

    suggestion = None
    if existing:
        scenario_index = scenario_selection_index(existing.scenario)
        intent_index = intent_selection_index(existing.intent)
    else:
        team_items, coverage_warnings = load_team_author_items(path.parent)
        for warning in coverage_warnings:
            st.warning(warning)
        suggestion = suggest_authoring_combination(team_items, role, item_id)
        scenario_index = scenario_selection_index(suggestion.scenario)
        intent_index = intent_selection_index(suggestion.intent)
    conversation_default = format_turns(existing.session_memory) if existing else ""
    message_default = existing.message.text if existing else ""
    language_default = existing.message.language_mix if existing else ["en"]
    terms_default = ", ".join(existing.message.local_terms) if existing else ""
    standard_default = existing.replies.standard if existing else ""
    local_default = existing.replies.local if existing else ""
    notes_default = existing.notes if existing else ""
    privacy_default = existing.privacy_ok if existing else False

    with st.form(f"author-{role}-{item_id}"):
        left, right = st.columns(2)
        with left:
            scenario = st.selectbox(
                "Scenario",
                SCENARIOS,
                index=scenario_index,
                placeholder="Choose a scenario",
                format_func=format_scenario,
            )
            if existing and scenario_index is None:
                st.caption(f"Existing label: `{existing.scenario}` — choose a canonical scenario before saving.")
            intent = st.selectbox(
                "Incoming message intent",
                INTENTS,
                index=intent_index,
                placeholder="Choose what the peer's message is doing",
                format_func=format_intent,
            )
            if existing and intent_index is None:
                st.caption(f"Existing label: `{existing.intent}` — choose a canonical intent before saving.")
            if suggestion:
                st.caption(f"Suggested from team coverage · currently used {suggestion.count} times")
            conversation = st.text_area(
                "Recent conversation",
                conversation_default,
                height=150,
                placeholder="peer: one prior message\nuser: your previous reply",
                help="Write 1–3 turns, one per line. Use peer:, user:, or other:.",
            )
            message = st.text_area(
                "Incoming peer message",
                message_default,
                height=100,
                help="This is the message that both suggested replies must answer.",
            )
            language_mix = st.multiselect("Incoming message language tags", LANGUAGE_TAGS, default=language_default)
            local_terms = st.text_input(
                "Local terms in the incoming message",
                terms_default,
                placeholder="Comma-separated, or leave empty",
            )
        with right:
            standard = st.text_area(
                "Standard reply",
                standard_default,
                height=120,
                help="A clear international-English reply to the incoming peer message.",
            )
            local = st.text_area(
                "Light local reply",
                local_default,
                height=120,
                help="English-led and restrained. Do not invent Mandarin or pile on slang.",
            )
            notes = st.text_area("Author notes", notes_default, height=100, placeholder="Optional reasoning for reviewers")
            privacy_ok = st.checkbox(
                "I confirm this item is fictional or fully rewritten and contains no identifying information.",
                value=privacy_default,
            )

        save = st.form_submit_button("Save draft", type="primary", disabled=not author_name.strip())

    if not save:
        return

    try:
        scenario = require_scenario(scenario)
        intent = require_intent(intent)
        turns = parse_turns(conversation)
        if not 1 <= len(turns) <= 3:
            raise ValueError("Recent conversation must contain 1–3 turns.")
        item = DialogueItem(
            id=item_id,
            split="unassigned",
            scenario=scenario,
            intent=intent,
            session_memory=turns,
            message=Message(
                text=message.strip(),
                language_mix=list(language_mix),
                local_terms=_split_terms(local_terms),
            ),
            replies=Replies(standard=standard.strip(), local=local.strip()),
            author=author_name.strip(),
            reviewer="",
            second_reviewer="",
            reviewer_notes="",
            second_reviewer_notes="",
            notes=notes.strip(),
            privacy_ok=privacy_ok,
            licence="team-authored",
            origin="team",
            status="draft_needs_team_review",
        )
        save_author_item(path, item)
    except ValueError as exc:
        st.error(f"Not saved: {exc}")
        return

    action = "Updated" if existing else "Created"
    st.success(f"{action} {item.id} in {path.name}.")


if __name__ == "__main__":
    main()
