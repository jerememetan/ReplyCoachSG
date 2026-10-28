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
all 20 pending at that check and export disabled.

## Human review completed

The user completed all 20 comparisons on 27 October 2026: **13 suitable alternatives,
7 unsuitable alternatives, zero skips**. All references were confirmed. These
are one reviewer's training labels, not a blind human evaluation or a sample of
independent survey participants. No model score supplied their labels.

The suitable alternatives become extra positives; they must not be treated as
hard negatives merely because they came from another conversation. None of
these 20 duplicates an existing weak pair, so this pilot corrects zero existing
weak labels. It adds 20 pairs to the original 1,140: **1,160 training pairs = 393
positives + 767 negatives**, of which 760 negatives remain algorithmic weak labels.
The underlying 190 training conversations and 300 validation pairs are unchanged.

The completed named review is frozen in a private local snapshot. Its canonical
SHA-256 is `0fb6a320219b77a28558454eb14c75707d74e5c8491e9f53e8f9d4b528e5f169`.
Only the stripped 20-pair sidecar is uploaded; name and free-text notes are not
included. The final training-pair SHA-256 is
`fd5beba4ce743df33e45cb86295b8045fc966b5314494f3fb40c1c9e9e32c943`.

Separate private experiment:

- [Kaggle notebook](https://www.kaggle.com/code/tanjingminjereme/replycoach-v6-20261027-reviewed01).
- [Private minimal dataset](https://www.kaggle.com/datasets/tanjingminjereme/replycoach-v6-training-20261027-reviewed01).
- Both privacy flags and the notebook's T4 request were read back through the
  authenticated official MCP. This run starts from the same pinned pretrained
  initialization as the original V6; it does not continue from an evaluated
  checkpoint. Abbreviation augmentation is disabled.

The initial submission was rejected because its title exceeded Kaggle's 50-
character limit. The accepted notebook has a shorter title/slug. The immutable
uploaded dataset receipt retains the original requested notebook ID; the local
`notebook_metadata_fix.json` records requested and actual IDs. Future packaging
shortens these titles and rejects oversized resource names before writing.

## Completed reviewed-pair GPU run

The private notebook completed version 1 on Tesla T4. All four epochs ran:
**292 optimizer updates total**, with **epoch 3 retained** (219 updates at that
checkpoint). Base initialization hashes match the original V6 exactly. The
original pair-file hashes also match; only the reviewed sidecar adds 20 pairs.
The GPU's saved 1,160-row audit file is byte-identical to the local reviewed
dataset. Abbreviation augmentation is absent from this run's manifest.

| Epoch | Training pair loss | Validation pair loss | Total updates |
| --- | ---: | ---: | ---: |
| 1 | 0.66980 | 0.62674 | 73 |
| 2 | 0.47029 | 0.49443 | 146 |
| 3 | 0.39857 | **0.45699** | 219 |
| 4 | 0.33550 | 0.55435 | 292 |

Training, validation and checkpoint saving took 15.98 seconds after model loading;
notebook queuing, installation and downloads are additional time. The best weak-
pair validation loss is lower than the original 0.48777 and augmented 0.55064.
This alone is not evidence of improved reply selection. The best epochs and hence
the selected checkpoints' update counts differ between runs; this is one small
exploratory training-data experiment with one seed.

The downloaded checkpoint is kept separately at
`artifacts/reranker_v6/kaggle_output_20261027_reviewed01/reranker_v6/best/`.
The title-length regression fix passed the full **406-test suite in 147.37
seconds**. Packaging checks verify private settings and empty notebook outputs.

## Completed matched reply-selection replay

The local scorer loaded the downloaded weights and verified their actual tensor
hash against the GPU manifest. It then replayed all 400 saved source proposals on
the same 50 validation conversations. The same 4,000 existing AI-persona ratings
were joined after selection. Rating, packet and source-export hashes matched;
**zero new judge calls** were made.

| Selector | Standard appropriateness /5 | Local appropriateness /5 | Standard would-send | Local would-send |
| --- | ---: | ---: | ---: | ---: |
| Original V6 | 3.528 | 3.878 | 55.4% | 68.6% |
| V6 + abbreviation augmentation | 3.442 | 3.926 | 54.6% | 70.6% |
| **V6 + 20 reviewed alternatives** | **3.508** | **3.930** | **55.4%** | **71.0%** |
| Fixed V5 | 3.666 | 4.110 | 61.2% | 76.8% |

Paired differences versus original V6, with descriptive 95% conversation-
bootstrap intervals (20,000 resamples, seed 425):

- Standard appropriateness: **-0.020 [-0.064, 0.012]**.
- Local appropriateness: **+0.052 [-0.110, 0.240]**.
- Standard would-send: **0.0 percentage points [-0.6, 0.6]**.
- Local would-send: **+2.4 percentage points [-2.4, 8.4]**.

The changes are small and intervals include zero. Lower weak-pair loss has not
established an overall reply-selection improvement. Fixed V5 still has higher
point estimates for both styles. The evaluation recommendation was to retain
the original installed V6 and keep this checkpoint as a separate experiment,
rather than select a winner from repeated validation runs. The user subsequently
requested the reviewed checkpoint for the demo; see the operational update below.

The new checkpoint changes three Standard and six Local choices. For example,
ITEM-E-025 Local changes from “ya, tuesday no class” to “okay, thursday is fine.”
when the peer asks about Thursday; AI mean 2.0→5.0. ITEM-C-055 Local now accepts
the proposed exhibition time instead of changing the activity to eating. There
are also regressions, such as ITEM-D-037 Local replacing “okay, see you at mac”
with “ya, reached alr. waiting at the entrance,” which repeats the peer's arrival
status rather than acknowledging it. Improvements and regressions are retained
in `reviewed_selection_changes_20261027.json`; neither category is omitted.

### Interpretation and limits

This pilot supplies 20 extra labels, including only seven newly confirmed
negatives, while 760 original negatives remain weakly labelled. The 13 suitable
alternatives demonstrate why high-scoring non-reference replies cannot all be
assumed wrong. This is valuable label-quality evidence, not proof that the entire
bank has been repaired or that the model architecture is the sole bottleneck.

Binary suitability supervision also permits several valid answers without
directly teaching their relative appropriateness scores. The current judge has
known questionable ratings; these comparisons are diagnostic. One reviewer's
training-label decisions are not a blind human evaluation. The validation set
also selects checkpoints using weak-pair loss, and repeated development
experiments, one seed, small sample size and different selected-update counts
limit causal/statistical claims. No held-out test results are reported here.

Raw evidence under `artifacts/reranker_v6/` includes
`replay_reviewed_20261027_01/result.json`, `reviewed_comparison_20261027.json`,
`reviewed_selection_changes_20261027.json` and the private reviewed-run folder.
The latter preserves the package receipt, metadata correction, privacy readbacks,
GPU history, replay log and 406-test output. The named completed review is frozen
locally in `hard_review/snapshots/`; the uploaded sidecar contains no reviewer
names or comments.

**Status:** human review, private GPU training, checkpoint verification and
matched development replay are complete. No test-set evaluation or claim of a
reliable improvement has been made.

## Demo selection update — 28 October 2026

At the user's explicit request, All models now defaults to **Fine-tuned V6
(20 reviews)**, using the separately downloaded reviewed checkpoint above.
Set `REPLYCOACH_V6_REVIEWED_CHECKPOINT` to override its location. **Fine-tuned V6
(original)** and **Frozen MiniLM** remain selectable. The original checkpoint
is preserved at `artifacts/reranker_v6/best/`. Missing reviewed weights produce
a visible error, with no silent substitution. This reversible demo choice
does not alter the measured comparison or establish a reliable improvement.

Verification: 67 affected demo/resource/ranking tests passed. A live request on
port 8511 compared all four sources, returned both styles, and displayed the
reviewed checkpoint revision `6e2664f06cc2c80fa1dcbeee72e24857e4b6178a11dc13e8716c5acafa69c4a4`.
The ignored local screenshot is `artifacts/reranker_v6/reviewed_demo_default_20261028.png`.
