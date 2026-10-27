# All-model reply selection: approved design

Date: 26 October 2026. Status: written design approved by the user on 26 October;
implementation verified on 27 October. No new training is proposed.

## Goal and scope

Add an **All models (experimental)** option to the existing Streamlit demo. The
user enters the recent conversation and the peer's incoming message once and
clicks **Suggest two replies**. BM25, MiniLM, V4 and V5 each propose one Standard
and one Local reply. A single frozen ranker chooses one final reply per style.

The first version compares **eight source proposals**, rather than every hidden
retrieval candidate. Each source retains its existing internal reranker/fallback
behaviour. This makes the experiment match the eight proposals per conversation
in the completed 4,000-rating evaluation. V4's source proposal may therefore be a
clearly labelled fallback. It must not be credited as an RNN generation.

Existing comparison, retrieval-only, generator-only and two-source Hybrid modes
remain available and keep their current selection policy. The new mode is
experimental and does not become the default based solely on being implemented.

## Chosen approach and alternatives

Use a pretrained cross-encoder without further training. The initial model is
`cross-encoder/ms-marco-MiniLM-L6-v2`, pinned to revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Its published licence is Apache 2.0.
It runs locally on CPU and does not call hosted inference or live Gemma judging.

The model was trained for MS MARCO passage relevance, not Singlish reply quality.
It is a candidate selection baseline, not an established conversational judge.
The validation experiment below decides whether it helps.

Alternatives considered:

- Retain the existing 0.6-input-score/0.4-overlap formula as a baseline. It is fast
  but unfair across sources whose score distributions differ.
- Train a task-specific reranker. Deferred because suitable labels, negative
  examples and tuning would create additional work within the user's one-week
  target. Existing AI ratings remain evaluation records, never training labels.

## Source collection and ranking

1. Prepare memory/normalisation once using the existing selected M0/M1/M2 mode.
2. Run each requested source once. Preserve its existing proposed Standard/Local
   pair, input scores, selected bank item, fallback origin and any error.
3. Treat each proposed reply as one candidate for its declared style. Group exact
   duplicate text within a style after normalising whitespace only. Preserve all
   contributing sources; never deduplicate across Standard and Local.
4. Score every unique proposal with the same frozen cross-encoder. Pair A is a
   fixed-format conversation and incoming message; pair B is the reply text.
   Do not pass model names, original scores, human reference replies, AI ratings,
   or author/reviewer information to the scorer.
5. Rank Standard and Local pools separately. Choose the highest shared score in
   each. Distinct-text score ties use a deterministic text order, independent of
   source. Duplicates share one score and report every contributing source.

The ranker query format is fixed before the experiment:

```text
Conversation (oldest first):
Peer: ...
You: ...
Incoming peer message: ...
User reply:
```

Use explicit You/Peer labels supplied by the user where available. Mark an
unlabelled turn as Unknown rather than inventing a sender. The requested style
controls the pool; do not assume this search model understands style instructions
or add keyword-based Singlish penalties. Existing model input formats remain
unchanged for reproducibility. In particular, V5's current plain-string API still
guesses alternating speakers; this known limitation stays visible in metadata.

Use a maximum pair length of 512 tokens, preserving the reply and prioritising
the incoming message and newest memory if source truncation is needed. Record
truncation. Empty, invalid or non-finite scorer outputs are errors, not scores.

The shared raw logit is a ranking signal, not a calibrated probability. Preserve
the old source scores in metadata; never blend them into this source-neutral
selection. Do not reuse the legacy 0.15 fallback threshold for these new scores.
The first version selects a valid proposal without claiming semantic confidence.
If no source proposal exists, show an explicit unavailable state with no selected
reply rather than inventing a model output.

## Architecture boundaries

- `shared_ranker.py`: pinned model loading, fixed pair formatting, batched
  inference, duplicate scoring, deterministic selection and score metadata.
- `all_models.py`: collect the four existing source outputs once, isolate source
  failures, retain provenance and construct the shared selection result.
- `demo_resources.py`: bounded cached ranker and up to two cached generators so
  switching V4/V5 does not repeatedly evict and reload V5 in all-model mode.
- `demo_app.py`: new mode, two final clickable replies, per-source proposals,
  source availability, elapsed time and expandable ranking details.
- A development-only offline replay command: run the frozen ranker against the
  completed proposal exports and compare selections with the existing ratings.

Prefer separate result types for all-model selection rather than silently
reinterpreting the existing three SYSTEM_MODES or changing their exports.
Keep the shared Candidate schema; identify the new selector in metadata.
Original BM25/MiniLM/V4/V5/fallback identities must remain recoverable even when
the selected candidate's source marks a reranking step.

## Failure handling and UI

All four successfully available: show **4/4 models compared**, two selected
reply buttons, and the source(s) of each. Display a compact source-proposal table
with all eight outputs and an expandable score/provenance section.

A source fails: retain the other source outputs and clearly show **Partial
comparison: N/4 models available**, with the failed source named. Selection among
the available proposals is allowed, but must not be presented as a complete
four-model result. Never relabel another model's output as the missing source.

The shared ranker fails: show the source proposals and a clear ranking error;
do not silently substitute the old heuristic and call it the shared ranker.
Offer retry through the existing interaction pattern. One source failing or
changing input clears obsolete selected replies. Clicking a reply still fills
the display-only composer and sends nothing.

Do not show raw scores as accuracy percentages. The existing default mode is
unchanged. Model weights may download once from the official model repository;
subsequent scoring is local. Cache resources rather than reloading per click.

## Validation experiment

Use only the existing 50 development validation queries and disjoint 190-item
reply bank. Never read or tune against the 60 locked test items.

First run a read-only replay over the exact four-source proposals already judged.
For every conversation/style, use the frozen ranker to choose one of those four
proposals and join its saved ratings by exact packet candidate identity. The
ranker never receives those ratings. This is **counterfactual selection analysis
on a fixed saved candidate pool**, not a new live-model quality evaluation.
When identical text has multiple saved records, reuse the lowest anonymous
candidate ID and retain every contributing source. Never choose the record with
the most favourable rating.

Compare with:

1. Always choose the saved V5 proposal.
2. Always choose the saved MiniLM proposal.
3. The existing legacy score formula applied across the same four proposals.

Report paired mean appropriateness and simulated would-send differences, source
selection frequency, failures, and elapsed ranking time. Resample conversations
together for descriptive intervals; retain the evaluator limitations already
documented. Do not label these AI judgments as human accuracy or use them to train.

Separately run real live-source smoke checks, including partial-source failures,
identical proposals, both styles, empty input and changed input. A small fixed
subset should reproduce existing proposals before applying the new ranker.
Document any mismatch rather than replacing saved evidence. No automatic
4,000-rating rerun is required to implement this experiment.

Promotion to default is a separate decision after inspecting the evidence. If
the pretrained scorer does not improve on a fixed V5/MiniLM baseline, keep it
experimental and report the negative result; do not tune a sequence of ad hoc
rules until it appears to win. No new training or live LLM judge is in scope.

## Runtime evidence already collected

The existing V5 provider loaded the supplied base and LoRA adapter locally and
reproduced all six reply texts for ITEM-A-017, ITEM-A-038 and ITEM-C-038.
Generation of both styles took 4.51, 6.44 and 6.00 seconds on CPU after a 50.92
second initial load. The environment's PyTorch build is CPU-only. These are
three-item smoke measurements, not full-dataset or production latency claims.
Full evidence: `roles/D_transformer/LOCAL_VERIFICATION.md`.

## Documentation and review

Keep a tracked work log of verification, changes, exact commands, experiment
results and unresolved limitations. Update the demo setup instructions and model
licence inventory when the implementation is verified. Preserve all existing
4,000 ratings, model weights, data files and user edits. Internal planning notes
remain under ignored `docs/superpowers/`; this shareable design is tracked so the
team can review the proposed method.

Official model source and licence:
https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2

Official cross-encoder documentation:
https://www.sbert.net/docs/cross_encoder/pretrained_models.html

## Self-review

- Scope is one experimental mode and one frozen scorer, with no new training.
- Candidate count and offline replay match the existing eight judged proposals.
- Partial results, fallback origins and score units are explicitly distinguished.
- No old threshold is applied to raw logits; no claims of style expertise are made.
- Data/evaluation boundaries and unchanged default behaviour are explicit.
- The new input preserves available labels for scoring without silently changing
  the already-evaluated source model formats.
- Implementation, replay and runtime verification are recorded in
  `docs/all_models_ranking_worklog.md`. The negative replay result keeps the mode
  experimental; it does not become the default.
