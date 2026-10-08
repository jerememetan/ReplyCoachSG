"""Interactive BM25/MiniLM comparison with a display-only chat preview."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from replycoach import demo_resources
from replycoach.io import ROOT
from replycoach.pipeline import ComparisonResult, compare_replies
from replycoach.reply_bank import load_reply_bank


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


def render_comparison(comparison: ComparisonResult) -> None:
    st.divider()
    st.header("Suggested replies")
    st.caption("Click a reply to fill the conversation preview.")
    columns = st.columns(2, gap="large")
    for column, method, label in zip(columns, ("bm25", "minilm"), ("BM25", "MiniLM")):
        with column:
            st.subheader(label, anchor=f"reply-model-{method}")
            st.caption("Word matching" if method == "bm25" else "Embedding similarity · all-MiniLM-L6-v2")
            if method in comparison.errors:
                st.warning(f"{label} unavailable: {comparison.errors[method]}")
                continue
            result = comparison.results[method]
            for style_label, candidate in (("Standard English", result.standard), ("Light local", result.local)):
                st.markdown(f'<p class="reply-style-label">{style_label}</p>', unsafe_allow_html=True)
                st.button(
                    candidate.candidate_text,
                    key=f"select_{method}_{candidate.style.lower()}",
                    use_container_width=True,
                    type="secondary",
                    on_click=select_reply,
                    args=(candidate.candidate_text,),
                )
                if candidate.source == "fallback":
                    st.caption("Fallback · no sufficiently ranked saved reply")
                else:
                    st.caption(f"Reranker · ranking score {candidate.score:.3f} · example {candidate.item_id}")
            with st.expander(f"Retrieved candidates · {label}"):
                st.dataframe(
                    [
                        {
                            "style": candidate.style,
                            "ranked by": "Reranker",
                            "ranking score": round(candidate.score, 3),
                            "retrieval score": round(candidate.meta["v6_input_score"], 3),
                            "example": candidate.item_id,
                            "reply": candidate.candidate_text,
                        }
                        for candidate in result.candidates
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
    st.caption("Scores are ranking signals, not accuracy percentages, and are not directly comparable between methods.")
    prepared = comparison.prepared
    st.write("**Normalised incoming message:**", prepared.normalised_message)
    st.write("**Selected memory:**", prepared.memory or "(none)")
    if prepared.detected_terms:
        st.write("**Detected local terms:**", ", ".join(t["surface"] + " → " + t["canonical"] for t in prepared.detected_terms))


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
        #reply-model-bm25, #reply-model-minilm { font-size: 1.75rem;
            border-top: 2px solid #3b82f6; padding-top: 1rem; }
        .reply-style-label { font-size: 0.82rem; font-weight: 600;
            letter-spacing: 0.03em; margin: 0.6rem 0 0.2rem; }
        /* Scope the chat-option treatment to replies, not the compare button. */
        [class*="st-key-select_"] button { min-height: 88px; padding: 1rem 1.25rem;
            border: 1.5px solid #3b82f6; border-radius: 1.1rem 1.1rem 0.3rem 1.1rem;
            background: rgba(59, 130, 246, 0.07); color: inherit; box-shadow: none;
            justify-content: flex-start; text-align: left;
            transition: background-color 160ms ease, border-color 160ms ease; }
        [class*="st-key-select_"] button > div { justify-content: flex-start; }
        [class*="st-key-select_"] button [data-testid="stMarkdownContainer"],
        [class*="st-key-select_"] button p { font-size: 22px; font-weight: 600;
            line-height: 1.45; text-align: left; white-space: normal; overflow: visible;
            text-overflow: clip; overflow-wrap: anywhere; margin: 0; }
        [class*="st-key-select_"] button:hover { background: rgba(59, 130, 246, 0.16);
            border-color: #60a5fa; color: inherit; }
        [class*="st-key-select_"] button:active { background: rgba(59, 130, 246, 0.24); }
        [class*="st-key-select_"] button:focus-visible { outline: 3px solid #3b82f6;
            outline-offset: 3px; border-color: #60a5fa; box-shadow: none; }
        @media (max-width: 640px) {
            [class*="st-key-select_"] button { min-height: 76px; padding: 0.85rem 1rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            [class*="st-key-select_"] button { transition: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Reply Coach")
    st.caption("Casual peer coordination for exchange students. Standard English + light local/Singlish.")

    st.session_state.setdefault("selected_reply", "")
    st.session_state.setdefault("last_comparison", None)

    with st.expander("Reply bank settings"):
        bank_choice = st.selectbox("Reply bank", ["Pilot examples (drafts)", "Reviewed development file"], key="bank_choice")
        development_path = ""
        if bank_choice == "Reviewed development file":
            development_path = st.text_input(
                "Development JSONL path", key="development_path",
                help="Only completed, independently reviewed dev records are allowed. Test and unassigned records are rejected.",
            )
        st.caption("MiniLM runs locally. The first comparison may download the public model weights.")
    bank = None
    try:
        if bank_choice == "Reviewed development file":
            if not development_path.strip():
                raise ValueError("Enter a reviewed development JSONL path, or select the pilot examples.")
            path = Path(development_path.strip())
            bank = load_reply_bank(path if path.is_absolute() else ROOT / path)
        else:
            bank = load_reply_bank()
        st.caption(f"Reply bank: {bank.label} · {len(bank.items)} examples · {bank.path.name}")
    except (OSError, ValueError) as exc:
        st.error(f"Cannot load reply bank: {exc}")

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        session_text = st.text_area(
            "Recent conversation",
            "Peer: lecture ending soon sia\nYou: yeah I'm starving",
            height=180,
            key="conversation",
            help="One message per line. Start a line with You: or Peer: to show its sender in the preview.",
        )
        message = st.text_input("Incoming message", "eh later go makan? can 吃饭 then library", key="incoming")
        memory_mode = st.selectbox("Memory", ["M2", "M1", "M0"], key="memory_mode")
        run = st.button("Compare reply suggestions", type="primary", key="compare", disabled=bank is None or not message.strip())

    identity = (bank_choice, development_path, bank.fingerprint if bank else None, session_text, message, memory_mode)
    if bank is None or st.session_state.get("comparison_identity") != identity:
        st.session_state.last_comparison = None
        st.session_state.selected_reply = ""
    st.session_state.comparison_identity = identity

    if run and bank is not None and message.strip():
        session = [parse_preview_line(line)[1] for line in session_text.splitlines() if line.strip()]
        st.session_state.selected_reply = ""
        with st.spinner("Comparing BM25 and MiniLM…"):
            st.session_state.last_comparison = compare_replies(
                message.strip(), session, memory_mode,
                lambda: demo_resources.get_bm25(bank.serialized),
                lambda: demo_resources.get_minilm(bank.serialized),
                bank_fingerprint=bank.fingerprint,
            )

    with right:
        render_preview(session_text, message, st.session_state.selected_reply)

    comparison = st.session_state.last_comparison
    if comparison is not None:
        render_comparison(comparison)
        if comparison.errors and st.button("Retry comparison", key="retry_comparison"):
            # Same identity as the displayed result; changed inputs clear it above.
            session = [parse_preview_line(line)[1] for line in session_text.splitlines() if line.strip()]
            with st.spinner("Retrying comparison…"):
                st.session_state.last_comparison = compare_replies(
                    message.strip(), session, memory_mode,
                    lambda: demo_resources.get_bm25(bank.serialized),
                    lambda: demo_resources.get_minilm(bank.serialized),
                    bank_fingerprint=bank.fingerprint,
                )
            st.rerun()


if __name__ == "__main__":
    main()
