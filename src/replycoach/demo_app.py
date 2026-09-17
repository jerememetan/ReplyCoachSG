"""Interactive Streamlit demo for Friday Week 4 / later polish."""

from __future__ import annotations

from html import escape

import streamlit as st

from replycoach.pipeline import ReplyCoach


def parse_preview_line(line: str) -> tuple[str, str]:
    """Return a display speaker and text from an optional You:/Peer: prefix."""
    cleaned = line.strip()
    for speaker in ("you", "peer"):
        prefix = f"{speaker}:"
        if cleaned.lower().startswith(prefix):
            return speaker, cleaned[len(prefix) :].strip()
    return "neutral", cleaned


def build_preview_html(conversation_text: str, incoming_message: str, selected_reply: str = "") -> str:
    """Build one safe HTML block so Streamlit keeps bubbles inside the preview."""
    fragments = ['<div class="chat-preview">']
    lines = [line for line in conversation_text.splitlines() if line.strip()]
    if not lines:
        fragments.append('<p class="empty-chat">Your recent messages will appear here.</p>')
    for line in lines:
        speaker, text = parse_preview_line(line)
        fragments.append(f'<div class="message-row {speaker}"><div class="bubble {speaker}">{escape(text)}</div></div>')
    if incoming_message.strip():
        fragments.append('<p class="incoming-label">MESSAGE TO REPLY TO</p>')
        fragments.append(
            f'<div class="message-row peer"><div class="bubble incoming">{escape(incoming_message.strip())}</div></div>'
        )
    composer_class = "selected" if selected_reply else "empty"
    composer_text = escape(selected_reply) if selected_reply else "Message to send…"
    fragments.append(f'<div class="preview-composer {composer_class}" role="textbox" aria-readonly="true">{composer_text}</div>')
    fragments.append("</div>")
    return "".join(fragments)


def render_preview(conversation_text: str, incoming_message: str, selected_reply: str = "") -> None:
    """Render a display-only conversation preview beside the editor."""
    st.markdown("#### Conversation preview")
    st.caption("Add `You:` or `Peer:` to show who sent each message. Press Ctrl+Enter in the conversation box to refresh.")
    st.markdown(build_preview_html(conversation_text, incoming_message, selected_reply), unsafe_allow_html=True)


def select_reply(reply_text: str) -> None:
    """Store a display-only reply for the preview composer."""
    st.session_state.selected_reply = reply_text


def main() -> None:
    st.set_page_config(page_title="Reply Coach", layout="wide")
    st.markdown(
        """
        <style>
        .chat-preview { min-height: 300px; padding: 1rem; border: 1px solid #d7dce5;
            border-radius: 0.75rem; background: #f7f8fb; }
        .message-row { display: flex; margin: 0.45rem 0; }
        .message-row.you { justify-content: flex-end; }
        .message-row.peer, .message-row.neutral { justify-content: flex-start; }
        .bubble { max-width: 82%; padding: 0.6rem 0.8rem; border-radius: 1rem;
            line-height: 1.4; color: #172033; }
        .bubble.you { background: #dbeafe; border-bottom-right-radius: 0.25rem; }
        .bubble.peer { background: #ffffff; border: 1px solid #d7dce5; border-bottom-left-radius: 0.25rem; }
        .bubble.neutral { background: #edf0f5; border-bottom-left-radius: 0.25rem; }
        .bubble.incoming { background: #fff2cc; border: 1px solid #e7be5f; border-bottom-left-radius: 0.25rem; }
        .incoming-label { margin: 1rem 0 0.2rem; color: #8a5a00; font-size: 0.72rem;
            font-weight: 700; letter-spacing: 0.06em; }
        .preview-composer { margin-top: 1.2rem; padding: 0.7rem 0.8rem; border-radius: 0.55rem;
            border: 1px solid #c5ccd8; background: #ffffff; color: #172033; }
        .preview-composer.empty { color: #667085; }
        .preview-composer.selected { border-color: #7aa7e8; background: #eff6ff; }
        .empty-chat { color: #667085; font-style: italic; margin: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Reply Coach")
    st.caption("Casual peer coordination for exchange students. Standard English + light local/Singlish.")

    if "coach" not in st.session_state:
        st.session_state.coach = ReplyCoach.from_pilot()
    if "selected_reply" not in st.session_state:
        st.session_state.selected_reply = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        session_text = st.text_area(
            "Recent conversation",
            "Peer: lecture ending soon sia\nYou: yeah I'm starving",
            height=180,
            help="One message per line. Start a line with You: or Peer: to show its sender in the preview.",
        )
        message = st.text_input("Incoming message", "eh later go makan? can 吃饭 then library")
        memory_mode = st.selectbox("Memory", ["M2", "M1", "M0"])
        run = st.button("Suggest two replies", type="primary")

    if run and message.strip():
        coach: ReplyCoach = st.session_state.coach
        coach.memory_mode = memory_mode
        session = [parse_preview_line(line)[1] for line in session_text.splitlines() if line.strip()]
        st.session_state.selected_reply = ""
        st.session_state.last_result = coach.suggest(message.strip(), session=session)

    with right:
        render_preview(session_text, message, st.session_state.selected_reply)

    result = st.session_state.last_result
    if result is not None:
        left, right = st.columns(2)
        left.subheader("Standard English")
        left.button(
            result.standard.candidate_text,
            key="select_standard",
            use_container_width=True,
            type="primary",
            on_click=select_reply,
            args=(result.standard.candidate_text,),
        )
        left.caption(f"{result.standard.source} · score {result.standard.score:.3f}")
        right.subheader("Light local")
        right.button(
            result.local.candidate_text,
            key="select_local",
            use_container_width=True,
            type="primary",
            on_click=select_reply,
            args=(result.local.candidate_text,),
        )
        right.caption(f"{result.local.source} · score {result.local.score:.3f}")
        st.write("**Normalised query:**", result.normalised_message)
        st.write("**Selected memory:**", result.memory or "(none)")
        if result.detected_terms:
            st.write("**Detected local terms:**", ", ".join(t["surface"] + " → " + t["canonical"] for t in result.detected_terms))
        st.subheader("Retrieved candidates")
        st.dataframe(
            [
                {
                    "style": c.style,
                    "source": c.source,
                    "score": round(c.score, 3),
                    "item": c.item_id,
                    "text": c.candidate_text,
                }
                for c in result.candidates
            ],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
