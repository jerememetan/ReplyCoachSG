"""Interactive Streamlit demo for Friday Week 4 / later polish."""

from __future__ import annotations

import streamlit as st

from replycoach.pipeline import ReplyCoach

st.set_page_config(page_title="Reply Coach", layout="wide")
st.title("Reply Coach")
st.caption("Casual peer coordination for exchange students. Standard English + light local/Singlish.")

if "coach" not in st.session_state:
    st.session_state.coach = ReplyCoach.from_pilot()

session_text = st.text_area(
    "Recent conversation (one message per line, newest last)",
    "lecture ending soon sia\nyeah I'm starving",
    height=120,
)
message = st.text_input("Incoming message", "eh later go makan? can 吃饭 then library")
memory_mode = st.selectbox("Memory", ["M2", "M1", "M0"])
run = st.button("Suggest two replies", type="primary")

if run and message.strip():
    coach: ReplyCoach = st.session_state.coach
    coach.memory_mode = memory_mode
    session = [line.strip() for line in session_text.splitlines() if line.strip()]
    result = coach.suggest(message.strip(), session=session)
    left, right = st.columns(2)
    left.subheader("Standard English")
    left.write(result.standard.candidate_text)
    left.caption(f"{result.standard.source} · score {result.standard.score:.3f}")
    right.subheader("Light local")
    right.write(result.local.candidate_text)
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
