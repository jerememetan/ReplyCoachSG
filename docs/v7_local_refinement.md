# V7 local reply refinement

Choose **All models (experimental)** in the Streamlit demo and click **Suggest
two replies**. V7 runs automatically after the existing four-source comparison:

1. BM25, MiniLM, V4 and V5 produce their usual Standard and Local proposals.
2. The existing shared ranker selects the original winner for each style.
3. V7 makes exactly three randomized, deduplicated variations of the Local winner.
4. The **same scorer instance**, with the same query format, weights and scoring
   logic, ranks the original plus those three variations and selects the final
   Local reply. The Standard English winner stays unchanged.

The **V7 local candidates** panel shows the four candidates, scores, selection,
and separate generation/reranking times. Reply buttons still fill the preview
without sending anything or repeating inference. Editing inputs clears results.
The JSON download keeps the source proposals and initial `ranked_candidates`,
adds the `v7` trace, and exports the displayed reply as top-level `local`.

V7 adds no settings, dependencies, training, generator calls, Gemma calls, or
evaluation criteria. Existing selectors and other system modes remain available.
`run_all_models` retains the original comparison behavior; the demo calls the
new `run_v7_all_models` wrapper.

## How the alternatives are made

The existing `data/lexicons/local_terms.json` supplies the terms. Enabled English
lexical, request, stance, negation and time entries can replace a whole canonical
word or phrase, preserving capitalization and surrounding text. Replacements do
not cascade. Disabled/ambiguous entries, identity mappings, location/status
mappings, and Chinese/pinyin entries are excluded. URLs, emails and inline code
are protected.

Each pool entry uses a single lexical replacement, a particle ending, or both.
The enabled JSON particles `lah`, `leh`, `lor`, and `sia` provide alternatives
when lexical matches are sparse. An existing standalone final particle is
replaced instead of stacked. A particle-like URL or email ending is preserved.
The pool is deduplicated case-insensitively after whitespace normalization; three
alternatives are sampled using fresh per-request randomness. The sampled choices
are retained in session state and the JSON trace.

The original is always eligible. An unavailable lexicon, insufficient distinct
alternatives, or failed/nonfinite reranker output produces a visible error,
retains the original Local winner, and enables retry. No smaller pool is silently
presented as a successful four-candidate V7 pass.

## Local measurements — 3 November 2026

Three handwritten conversations ran through the actual Streamlit app with the
installed V4/V5 models and the default Qwen + DailyDialog reranker. The bank was
the existing 20-example pilot; the held-out test was not read or evaluated.
The check used Windows, PyTorch 2.14.0 CPU, two PyTorch threads, Streamlit 1.65.0,
and Qwen's existing batch size of two. Public weights were already cached and
the run used offline loading. These are local observations from three requests,
not a general latency guarantee or a matched end-to-end benchmark.

| Handwritten case | Initial unique pairs | V7 lexicon time | Added V7 reranking | Full request |
| --- | ---: | ---: | ---: | ---: |
| Meeting confirmation, first use | 6 | 15.31 ms | 14.51 s | 116.04 s |
| Dinner invitation, warm models | 7 | 32.14 ms | 11.13 s | 94.61 s |
| Delayed train, warm models | 8 | 20.66 ms | 9.16 s | 96.14 s |

The additional ranking pass had a **median of 11.13 seconds**. Each request scored
exactly four extra pairs, using two additional Qwen batches and the already loaded
scorer. It loaded no additional model. The first full request includes model
loading; source generation and initial ranking also varied with the conversation.
A separate 300-call warm lexicon-only check had a median of **0.60 ms**; the app
stage timings above are the observed request-level figures.

| Case | Original Local winner | Final V7 reply |
| --- | --- | --- |
| Meeting | Yep, I checking at 7pm. | Yep, I checking at 7pm sia. |
| Dinner | Can, dinner first. | Can, dinner first lah. |
| Delay | Can, 7.15pm better. | Can, 7.15pm better lah. |

All three selected a variation and retained the parent's facts/times in these
examples. These winners contained no eligible lexical replacement, so the live
checks exercised the particle path; unit tests also exercise word/phrase changes.
The meeting example retains awkward source grammar. Lexicon changes do not fix
all generation problems, and particles can change tone. This smoke check does
**not** establish improved authenticity, naturalness, or overall reply quality.
The existing reranker is still allowed to select the original.

The local evidence is in `output/v7/real_model_receipt.json`, the three per-case
JSON files, and `output/v7/demo.jpg`. The screenshot shows the actual demo seeded
with the saved real-model meeting result; the three live AppTest requests verified
generation, both scoring passes, checkpoint identity, scorer reuse, four distinct
candidates, and click-to-fill without repeated inference. Raw evidence and the
local verification helpers are ignored by Git.

## Regression coverage

The final full suite passed: **473 passed, 1 warning in 195.88 seconds**. This
includes 30 new V7 unit/integration checks. Inference imports were initialized
before AppTest to keep first-use imports outside its existing 20-second request
timeout. Test temporary files were placed outside the repository so the existing
external-path privacy test exercised an external path. The warning concerns
pytest assertion rewriting for the already imported `anyio` dependency.
`git diff --check` also passed.

Tests cover exact candidate counts and deduplication, sparse matches, existing
particles, capitalization, phrase boundaries, URLs/emails/code, disabled terms,
randomness, unchanged initial proposals/selection, one source call each, scorer
reuse, original/variation winners, provenance and JSON, partial sources, visible
failure/retry, unchanged settings, and stable click-to-fill behavior.

The implementation changes `lexicon_refinement.py`, `all_models.py`, the demo
renderer/app, and the shared source tag list. Generator code, both existing
rerankers, the lexicon JSON, model artifacts, and Gemma evaluation remain unchanged.
