# Reviewed difficult reply comparisons

User-approved next experiment, 27 October 2026: prepare difficult alternatives
for training conversations, review their suitability, then train and compare
against the original V6 on unchanged validation data. The previous no-AI-label
rule remains: neither current reranker scores nor Gemma ratings supply labels.

## Scope and flow

Start with 20 comparisons, balanced across Standard/Local. Mine a BM25 shortlist
of other **training** items' human-authored replies, excluding both known replies
for the source item. Score those alternatives and the reference with the current
V6. Prioritize alternatives the model prefers over the reference; scores only
select the review queue, never mark an alternative unsuitable. Spread the queue
over training conversations and show the exact M2 context to the reviewer.

Use a new Reranker comparisons mode in the existing team reviewer app. A named
reviewer confirms the reference still fits, then labels the alternative suitable,
unsuitable, uncertain/skip, or reference-needs-correction/skip. Save directly to
an ignored review file with stale-update checks and a file lock. No edits to the
190/50/60 source datasets. The pilot is a targeted label collection, not another
full team review round or a human evaluation of held-out model quality.

Export only after all queue items have a named decision; skips do not become
training rows. Include suitable alternatives as additional positives. If a
reviewed alternative already occurs among weak negatives, replace that weak
pair rather than training contradictory labels. Bind exports to source IDs,
exact training queries/replies and original bundle fingerprints/hashes. Strip
reviewer names/comments from the private Kaggle payload; retain the local review
file hash and decision counts for audit. Refuse validation/test IDs and drafts.

Integrate a separate optional reviewed-pairs sidecar with the existing portable
trainer and private Kaggle packager. Keep this experiment separate from spelling
augmentation so there is one training-data change to assess. Preserve the
existing checkpoint and unaugmented/augmentation runs. Do not launch a new GPU
job before actual review decisions exist. The original validation/rating replay
remains evaluation-only, and the held-out test items remain unopened.

## Verification

Test training-only mining, known-positive exclusion, deterministic sampling,
pending/invalid/foreign review refusal, stale saves, valid alternatives replacing
weak negatives, unchanged validation and real optimizer consumption. Verify the
new reviewer mode with synthetic saves and a read-only check on the actual queue.
Verify the standalone private package; document the actual queue and next human
step. A prepared queue is not a completed review or trained checkpoint.

## Prepared pilot and use

The actual queue is `artifacts/reranker_v6/hard_review/review.json`: **20 distinct
training conversations, 10 Standard and 10 Local comparisons**, all initially
pending. No label was inferred from the scorer's preferences. BM25 proposes up
to four distinct training donors per reference; the installed trained V6 ranks
them. Prioritization uses alternative-minus-reference logit, alternates styles
and avoids repeating a source conversation. This is model-confusion sampling,
not automatic semantic error classification or a claim that all alternatives
are unsuitable. Some selected alternatives clearly appear valid.

Examples for review include ITEM-A-023 (apology versus offering to place an
order), ITEM-B-058 (printer-location question versus asking the peer to print),
and ITEM-A-018 (requested location change versus proposing next weekend).
ITEM-C-047 has a seemingly suitable alternative accepting the travel plan; the
reviewer can keep it as another positive. These are training examples, not the
diagnosed validation items.

Open the [local reviewer](http://127.0.0.1:8513/?review_task=reranker), or launch:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/replycoach/team_review_app.py --server.port 8513 --server.fileWatcherType none
```

Choose **Reranker comparisons** in the sidebar. Enter your name, read the exact
M2 conversation, confirm the reference if appropriate, label the alternative
and click **Save and next**. Both replies may be suitable. Use **Uncertain —
skip** for debatable alternatives or **Reference needs correction — skip** if
the existing reference does not fit. Notes are optional. These skips exclude
the new comparison, not the existing reference from the baseline dataset;
reference corrections remain a separate source-data issue to resolve.

Once every comparison has a decision, **Prepare reviewed training file** becomes
available if at least one suitable/unsuitable label exists. It writes a fresh
`reviewed_pairs_<review-hash>.json` beside the queue, with names/comments stripped.
Save this sidecar with the matching original bundle. Do not upload `review.json`.

To reproduce mining or export from a completed queue:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_reranker_review.py --output artifacts/reranker_v6/new_hard_review/review.json --count 20
.\.venv\Scripts\python.exe scripts/prepare_reranker_review.py --export-review artifacts/reranker_v6/hard_review/review.json --output artifacts/reranker_v6/hard_review/completed_pairs.json
```

Commands refuse existing outputs. The exporter rejects pending decisions and
requires the original source bundle to match. A stale/different review cannot
silently replace a displayed decision; the app offers **Reload comparison**.

## Training integration ready for completed decisions

The existing trainer accepts `--reviewed-pairs <completed-sidecar.json>`. The
packager accepts the same flag, validates the payload before creating output,
includes only the stripped sidecar, and generates a notebook with
`USE_REVIEWED_PAIRS = True`. Its training cell uses the human decisions and
disables abbreviation augmentation for this experiment. Both CLI and trainer
reject combining these two additions. Earlier training defaults remain usable.

```powershell
.\.venv\Scripts\python.exe scripts/package_reranker_kaggle.py --bundle artifacts/reranker_v6/bundle_20261027_01 --reviewed-pairs artifacts/reranker_v6/hard_review/completed_pairs.json --output artifacts/reranker_v6/new_reviewed_package --owner tanjingminjereme --tag 20261027-reviewed01
```

The reviewed additions can include multiple suitable replies per source/style.
A matching weak pair is replaced, including a 0→1 correction when a previously
mismatched reply is human-confirmed suitable. No original reference positive is
overwritten. The final training-pair payload and its hash, reviewed counts,
weak-label corrections and source-review hash are recorded in the output and
training manifest. Validation remains the original 300 weakly supervised pairs;
that checkpoint-selection metric itself remains a limitation. The matched
saved-proposal replay is still necessary before claiming improved selection.

Software checks: **404 tests passed in 117.78 seconds**. Tests cover explicit
review/skip decisions, valid-alternative correction, draft/foreign/tampered-data
refusal, stale saves, UI persistence and actual optimizer consumption. A separate
standalone package also completed two CPU optimizer updates on clearly synthetic
fixtures, marked nonproduction. Its private evidence is in
`artifacts/reranker_v6/hard_review/software_smoke/`. No synthetic decisions were
inserted into the actual 20-item queue. The real UI was checked read-only, with
all20 pending at that check and export disabled.

**Status:** queue and software are prepared; real human decisions are required
before a reviewed-pair export, GPU training or quality comparison can be
reported as complete. The current trained V6 checkpoint remains installed.
