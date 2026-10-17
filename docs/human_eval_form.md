# Shared output evaluation form and optional manual sanity check

Updated 17 October 2026. The initial evaluation uses ten persona-based AI
reviewers, not human survey participants. This form is retained for optional
manual sanity checks using the same rubric. E's `scripts/judge_replies.py`
automates the form with local Ollama and `data/evaluation/personas.csv`.
See `split_and_rating_protocol.md` and `../scripts/data-split/rating_rubric.md`.

Rate each system output independently. Do not look at system names. Order is
randomised during E's packet preparation. This task rates model replies; it does not edit the authored
dataset or ask reviewers to identify human-written text.

Conversation id: __________  
Rater: __________  
Date: __________

Incoming message:

>

Conversation history (speaker-labelled; the same history for every system):

>

| Output | Appropriateness 1–5 | Naturalness 1–5 | Style fit 1–5 | Would you send it? Y/N | Notes |
| --- | --- | --- | --- | --- | --- |
| A STANDARD |  |  |  |  |  |
| A LOCAL |  |  |  |  |  |
| B STANDARD |  |  |  |  |  |
| B LOCAL |  |  |  |  |  |
| C STANDARD |  |  |  |  |  |
| C LOCAL |  |  |  |  |  |

Systems A/B/C are a random permutation of retrieval-only, generator-only, and hybrid.

Criteria:

- Appropriateness: does it respond sensibly to the incoming message and context?
- Naturalness: does it sound like a fluent, plausible casual message?
- Style fit: does it match the requested Standard or Light local style?

Use the criterion-specific 1–5 anchors in the shared rubric. Naturalness replaces
the original usefulness criterion. "Would you send it?" is required boolean
feedback in the AI runner and optional in manual checks, not a fourth 1–5 criterion.
The summary's send percentage is simulated willingness, not surveyed human approval.

STANDARD should sound like plain neutral English; ordinary casual phrasing is
allowed. LOCAL should be English-led and only lightly local. Do not demand extra
slang when it would be forced, or reward invented Mandarin. Note problem flags and
any uncertainty separately from the scores.
