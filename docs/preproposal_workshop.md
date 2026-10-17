# Pre-proposal workshop (29 Sep 2026, ~15 min)

Not graded. Get feedback, then freeze the 3-page proposal.

Historical workshop outline. The original human-rating plan below was superseded
on 17 October by persona-based LLM evaluation: ten initial configurable AI personas
scoring appropriateness, naturalness and style fit. See `split_and_rating_protocol.md`.

## Suggested slide order

1. Problem in one sentence: mixed messages, two sendable replies.
2. Demo (30 s): run `python -m replycoach.demo_cli` or Streamlit. Show STANDARD vs LOCAL.
3. Architecture diagram from `docs/architecture.md`.
4. Locked decisions (no API, V4 vs V5, restrained local style, team data).
5. Data plan: 20 pilot fixtures now; 300 reviewed team items split into 240 development and 60 test; DailyDialog for RNN only.
6. Experiment journey V0–V6 and M0–M2.
7. Evaluation: blind human ratings, three systems.
8. Risks: tiny data, V5 may lose, slang temptation.
9. Roles and Friday artefacts.
10. Ask the instructors: is restrained Singlish in-scope? any licence concern with DailyDialog?

## Who speaks

A motivation + data; B memory/retrieval; C V3/V4; D V5 constraint; E demo. Keep a 2-minute buffer for questions.
