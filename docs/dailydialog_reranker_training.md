# DailyDialog → reviewed-data Qwen reranker

## Verified completed training — 29 October 2026

The user supplied `qwen_dailydialog_03_results.zip`; **both GPU training stages
completed successfully**. The imported results and checkpoints are preserved
under ignored `artifacts/reranker_qwen/dd_result_20261029/`. Previous checkpoints
remain unchanged. The [matched reply-selection comparison](reports/dailydialog_reranker_comparison_20261029.md)
is now complete: 3.634 Standard / 3.988 Local versus previous Qwen's 3.640 / 3.946
and V5-only's 3.666 / 4.110. This mixed development result does not establish a
quality gain; both style-specific appropriateness intervals against each requested
baseline include zero. The user subsequently selected this checkpoint as the
final V6 demo direction: **All models now defaults to V6 (Qwen + DailyDialog)**,
with the older MiniLM scorers still selectable. This deployment choice is separate
from the evaluation conclusion. See the README for private files and setup.

### Demo integration verified — 29 October 2026

The All models selector now defaults to **V6 (Qwen + DailyDialog)**. It lazily
loads the local pinned base and final stage-2 adapter, validates existing training
provenance and hashes, and caches the scorer. The previous reviewed/original
MiniLM checkpoints and frozen MiniLM remain selectable. Missing files or failed
provenance checks remain visible errors; proposals are preserved without a silent
ranker substitution. Switching selectors clears old selections and preview text.

Verification: **75 targeted demo/resource/shared-ranking/Qwen/all-model tests
passed**. A real Streamlit AppTest request used all four actual source models,
the 190-item training reply bank and the final adapter SHA-256
`1a92ef197458b5e9f1280612728849bc1048dcb1d0696bf5c2f3505c61f1ab66`.
For the Harbourfront/7pm confirmation example it selected Standard
“Yes, 7pm is fine.” and Local “Yep, 7pm is fine.” Both result exports retained
the adapter identity and `yes_minus_no_logit` score kind. Clicking the Standard
reply filled the display-only preview. The check completed in 68.3 seconds,
including 25.9 seconds in shared ranking with initial Qwen loading; these are
one local CPU run, not a latency guarantee or a new quality evaluation.
Private evidence: `artifacts/reranker_qwen/demo_smoke_20261029.json` and its log.
No new training, judging or held-out test access occurred.

| Stage | Training / validation pairs | Completed / maximum epochs | Selected epoch | Best validation loss | Successful updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| DailyDialog adaptation | 12,000 / 1,200 | 2 / 2 | 1 | 0.249795 | 1,500 |
| Reviewed refinement | 1,160 / 300 | 4 / 12 | 2 | 0.341845 | 292 |

Adaptation validation loss rose from 0.249795 at epoch 1 to 0.263652 at epoch 2,
while training loss fell from 0.300086 to 0.235613. Refinement validation loss was
0.370391, 0.341845, 0.344872 and 0.350246 across epochs 1–4. Patience 2 stopped
refinement after two nonimproving checks, keeping epoch 2. These results support
the selected checkpoints within the observed run; they do not establish that no
later epoch or different training configuration could improve.

Final checkpoint: `stage2_reviewed/best/`, initialized from **adaptation epoch 1**.
Its selected refinement epoch represents 146 updates; 292 updates were executed
before early stopping. The selected stage-1 checkpoint represents 750 updates,
although adaptation executed 1,500 updates. Recorded training-loop elapsed times
were approximately 56.2 and 11.1 minutes, excluding preparation/base loading.

The overflow correction was exercised once in adaptation: after 311 successful
updates, the window starting at training row 4,976 produced nonfinite gradients.
The optimizer did not update, the loss scale fell from 65,536 to 32,768, and the
same window was retried successfully. No further adaptation overflow and no
refinement overflow were recorded. This verifies recovery in the supplied real
T4 run, beyond the earlier CPU regression checks.

Integrity verification passed: ZIP CRCs and safe paths, both adapter hashes,
112 finite LoRA tensors per adapter, the local pinned Qwen base-weight hash,
prompt/revision identity, parent adapter file/parameter/metadata hashes, serialized
training-pair hashes, external split hashes and disjoint dialogue identities.
The final training pairs byte-match the approved reviewed bundle, and the project
validation fingerprint/pair hash match the unchanged 50-conversation bundle.
No project test items or AI evaluation ratings were introduced by this run.

| Same project-validation pair-loss comparison | Selected validation loss |
| --- | ---: |
| Reviewed-only Qwen, original four-epoch run | 0.349055 |
| Reviewed-only Qwen, maximum-12 follow-up | 0.349164 |
| DailyDialog → reviewed Qwen | 0.341845 |

The new selected loss is 0.007210 lower than the original reviewed-only run, on
the same training/validation pair identities. This is a small descriptive
development result, not a significance test or evidence that chosen replies are
better. More data and updates, plus the corrected numerical policy, differ between
protocols; do not attribute all changes solely to the dataset source. Stage-1 and
stage-2 losses use different validation data and must not be compared directly.

The completed comparison replays the same 400 saved validation proposals / 50
conversations and compares 100 selected replies with reviewed-only Qwen, raw Qwen,
reviewed MiniLM and fixed V5. Only ten replies changed versus previous Qwen;
existing AI appropriateness ratings rose for five, fell for four and tied for one.
The repeated-speaker subgroup has five conversations and all ten choices remain
unchanged. Duplicate-text sensitivity leaves the conclusion unchanged. Reused
evaluation records remain evaluation-only; no judge-derived training labels or
project test items were used. See the linked report for intervals, examples and
the charger-role error incorrectly rated 5/5 by the existing judge.

Private evidence: `experiment.json`, both histories and `best/qwen_training.json`
files, `training_curves.png`, `stage1_dailydialog/overflow_events.jsonl`,
`adaptation_pairs/manifest.json` and `verification_receipt.json` in the imported
result directory. Source ZIP SHA-256:
`3fab55c6aece76f76e5a8a5bf2bce8813f5ce05ade3ba2ef44bda71213c5a780`.

## Original setup and correction history

28 October 2026. User-approved design and implementation checklist. This prepares
code and a private input bundle for the user to run on Kaggle; no GPU job is
submitted and no checkpoint is promoted by this change.

**Kaggle setup completed:** [private notebook](https://www.kaggle.com/code/tanjingminjereme/replycoach-dailydialog-qwen)
version 1 is saved with the [private input dataset](https://www.kaggle.com/datasets/tanjingminjereme/replycoach-qwen-pairs-20261028-20)
attached, T4 GPU selected and Internet enabled. It was saved using Quick Save
(`kernels push --no-run`); training had not started at that setup point. Open the notebook, choose
Edit, then Save Version → Save & Run All to start the saved background training
run. Run All in the editor is also available for an interactive session.
The uploaded data was downloaded again and every mounted file hash verified;
the saved notebook source, private visibility and input attachment were checked.
Kaggle consumes `dataset-metadata.json` as upload settings, so the two-stage
notebook's integrity receipt correctly excludes that non-mounted control file.

## 29 October: FP16 gradient overflow correction

The user's interactive run `qwen_dailydialog_02` reached at least 300 successful
updates in adaptation epoch 1, then stopped in `clip_grad_norm_` because gradients
were nonfinite. The outer notebook's `CalledProcessError` is the script exit,
not a separate data/setup failure. That epoch had not reached its checkpoint
selection, so this run cannot resume those updates with the current notebook.

Root cause confirmed in the training loop: strict clipping ran immediately after
AMP unscaling and raised before `GradScaler.step/update` could skip the invalid
update and reduce the loss scale. A real FP16-backward overflow was reproduced
on CPU with an enabled scaler. This establishes the missing recovery path; it
does not independently identify why that particular T4 batch first overflowed.

The correction checks gradients after unscaling. Nonfinite scaled gradients cause
AMP to skip the optimizer update and reduce its scale; the same accumulation
window is retried, up to eight retries. Successful finite windows still use
`error_if_nonfinite=True` for clipping. Nonfinite loss, invalid unscaled gradients
without a scaler, or persistent overflow remain fatal. No learning-rate, data,
epoch-cap, patience, model, prompt or LoRA changes are bundled into this fix.

Logs now include `qwen_gradient_overflow` events with old/new loss scales,
`overflow_events.jsonl`, retry totals in histories and checkpoint metadata, and
the `finite_only_amp_retry_v1` gradient-policy identity. Counters and average loss
include only successful update windows, not failed attempts. Failed attempts
cannot update Adam state or weights. Replayed examples are identical; training
dropout draws need not be identical on a retry.

The revised notebook checks the input package's policy identity before starting,
defaults to fresh output `qwen_dailydialog_03`, and preserves earlier results.
After the input update, reopen the saved notebook and update the attached input
to the latest dataset version if Kaggle still mounts an older version. Restart
with a new run name; do not edit `error_if_nonfinite` or read `experiment.json`
as though the failed run completed. At that correction stage, a GPU rerun was still needed. The completed rerun
is verified above; reply-quality evaluation is pending.

Regression checks cover genuine half-precision overflow/backoff without weight
updates, subsequent successful recovery, same-window replay and correct step
counts, persistent-failure bounds, and strict failure without loss scaling.
Reference: [PyTorch AMP gradient clipping and step-skipping](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html#gradient-clipping).

Deployment verified: the corrected private input was uploaded as a new dataset
version, and notebook **version 3** was saved using `--no-run`. Its default output
is `qwen_dailydialog_03`. The uploaded files were downloaded and hash-checked, and
the saved notebook's source, input attachment, private visibility, GPU and Internet
settings were verified. All 27 focused regression checks passed. Training was not
started by that deployment step. The user subsequently completed the T4 run,
whose imported artifacts are verified above. Local package/evidence: `artifacts/reranker_qwen/dd29/`.

## Design

Keep the pinned Qwen3-Reranker-0.6B yes/no suitability objective and LoRA settings.
Stage 1 adapts a fresh adapter to observed DailyDialog next-turn replies versus
deterministically mismatched replies from different training dialogues. Stage 2
continues **the selected stage-1 adapter**, with a fresh optimizer, on the existing
190-conversation training bundle including the 20 reviewed alternatives. Each
stage has a separate output folder, best checkpoint, history and provenance.

The DailyDialog paper describes two-party speaker turns. Its public release has
no explicit speaker IDs: alternating **utterance turns**, not sentences, are a
documented source-format assumption. Map roles relative to the candidate speaker:
the incoming turn is Peer, the candidate is You, and earlier turns retain their
relative roles. Do not split sentences or alternate the project's actual messages;
the reviewed bundle already preserves consecutive Peer/You messages unchanged.
This assumption is not independently verified speaker annotation.

Use only the pinned public **train.zip**, with a fixed archive hash; download no
DailyDialog test split. Deduplicate complete dialogues, partition them by dialogue
hash into adaptation train/validation, and exclude exact project-text overlaps.
Stage 1 selects on its own DailyDialog validation pairs; stage 2 selects on the
unchanged 50-conversation project validation bundle. Never load the 60 project
test items. Observed replies are weak positives, not quality gold; mismatches
can be valid replies. No generated dialogue or AI judge labels enter training.

## Implementation plan

- [x] Test role-relative context, whole-dialogue separation, deterministic donors,
  overlap filtering, archive parsing and private allowlisted packaging.
- [x] Add a portable DailyDialog preparation module and two-stage CLI.
- [x] Reuse the existing optimizer loop; add verified adapter continuation and
  stage provenance without changing original one-stage defaults.
- [x] Generate an importable Kaggle notebook and a minimal private upload folder.
- [x] Run focused data tests and tiny-model optimizer/continuation checks locally;
  validate notebook code and the real private bundle. Do not start GPU training.
- [x] Document upload/run/download steps, limits and final-report evidence.

## Run it on Kaggle

Prepared local files (ignored, containing the actual reviewed training bundle):

- Input ZIP: `artifacts/reranker_qwen/dd20/private_input.zip`.
- Importable notebook: `notebooks/E_qwen_dailydialog_kaggle.ipynb`.
- Original packaged notebook: `artifacts/reranker_qwen/dd20/notebook/Qwen_reranker.ipynb`.
- Corrected 29 October package: `artifacts/reranker_qwen/dd29/` (same remote dataset
  and notebook IDs, with the new trainer and regenerated integrity receipt).

1. In Kaggle, create a **private dataset** from `private_input.zip`. Ensure the ZIP
   contents are extracted, so `manifest.json` and the Python files are visible.
   The receipt's suggested dataset slug is optional when using the UI.
2. Create a notebook, use **Import Notebook**, and select the `.ipynb` above.
3. Add that private dataset as input. Select a GPU (one T4 is sufficient) and turn
   Internet on. Keep the notebook private.
4. Run All, or Save Version with **Save & Run All** for a saved background run.
   No Kaggle credentials go into the notebook. Do not close an interactive session
   expecting it to become a saved background job automatically.
5. The notebook installs the same package versions as the completed Qwen experiment,
   checks every supplied file's hash, downloads the pinned DailyDialog archive and
   runs the two stages. It prints progress every 20 optimizer updates and each epoch.
6. Download `<RUN_NAME>_results.zip` from the last cell or Kaggle's Output
   panel. Keep the complete results ZIP; no base-model weight download is needed
   from Kaggle. Final adapter: `stage2_reviewed/best/`.

The settings cell exposes `MAX_TRAIN_QUERIES` (4,000), adaptation validation
queries (400), adaptation epochs (2 maximum, patience 1), and refinement epochs
(12 maximum, patience 2). Each query yields one observed positive and two weak
negatives. Effective batch size is 16: microbatch 2 × accumulation 8. Both stages
use AdamW 2e-5, rank 8 LoRA on q/v projections, alpha 16, dropout 0.05, maximum
512 tokens, seed 425 and frozen base weights. Instruction and candidate reply
are retained; oldest query tokens may be truncated by the existing scorer.

Stage 2 uses the 1,160 reviewed/weak training pairs and unchanged 300 validation
pairs from the existing bundle, representing 190 / 50 project conversations.
It loads stage 1's **best adapter**, not the last epoch and not a new adapter.
Parameter and file hashes verify this handoff. The optimizer resets between
stages; this is staged fine-tuning, not optimizer-state resumption.

Changing `RUN_NAME` creates a separate result folder. Existing output directories
cause an explicit error; they are never deleted or silently reused. If a session
fails midway, preserve its available outputs, but this notebook does not implement
mid-epoch resumption. The original adapter and demo default remain untouched.

## Regenerate the private package after code changes

From the repository root, use a fresh output directory and run tag:

```powershell
.\.venv\Scripts\python.exe scripts/package_dailydialog_qwen.py `
  --bundle artifacts/reranker_qwen/kaggle_20261028_12/dataset `
  --output artifacts/reranker_qwen/dd21 `
  --owner tanjingminjereme --tag 20261028-21
```

Upload the files inside the generated `dataset/` folder; they are allowlisted
source code, the train/validation manifest and pairs, the minimal 20-review
supplement and integrity metadata. No held-out project test items, reviewer
names, free-text review comments, AI ratings, credentials or model checkpoints
are packaged. ZIP that folder if using Kaggle's web upload. The generator does
not create remote resources or launch a run.

## Verification performed locally

- Focused tests cover roles relative to the reply speaker, deterministic
  whole-dialogue splitting, training-only negative donors, normalized duplicate
  and project-overlap exclusion, unusable-turn handling, archive hash rejection,
  standalone packaging and notebook compilation.
- A tiny Qwen architecture completes both optimizer stages on CPU test fixtures,
  confirms adapter continuation, unchanged frozen weights and unchanged source
  bundle bytes. These fixtures/checkpoints are marked nonproduction and are
  never included in the actual private training package.
- Data-only preparation against the real pinned archive and actual private bundle:
  11,118 source dialogues; 858 duplicates, 625 project-overlap dialogues and one
  punctuation-only-turn dialogue removed. Remaining adaptation partition:
  8,671 train / 963 validation dialogues; sampled pairs: 12,000 / 1,200.
- The supplied upload is approximately 0.62 MB uncompressed; package hashes and
  standalone CLI imports were checked. These checks preceded the user GPU run, now verified above. Matched reply-selection
  evaluation is still pending; no improvement in reply quality is claimed.

## Evaluation and final report

Record both stages' actual completed epochs, selected epochs, pair counts,
losses, package versions, archive/prompt/adapter hashes, runtime and training curves
from `experiment.json`, `history.json` and the generated plot. Compare raw Qwen,
reviewed-only Qwen, this two-stage adapter and V5 against the same saved project
validation candidate pool. Report Standard and Local separately, including
examples with consecutive same-speaker messages. Do not equate lower pair loss
with better reply selection. No demo promotion or final-test evaluation is automatic.

Limitations: DailyDialog is written English-learning dialogue, not authentic
Singapore texting; alternating source turns are an assumption without explicit
speaker IDs; observed replies are weak positives; cross-dialogue mismatches may
be valid alternatives. Exact text exclusion does not detect paraphrase overlap.
Whole-dialogue splitting does not eliminate every repeated phrase or partial
conversation across dialogues. Do not manufacture LOCAL translations or extra
speaker-labelled messages to increase dataset size. Project data retains its
original roles; DailyDialog alone does not teach same-speaker message bursts.

## Sources

- [DailyDialog paper](https://aclanthology.org/I17-1099.pdf), section 2.1:
  human-written English-learning dialogues, more-than-two-party dialogues removed.
- [Author dataset card and CC BY-NC-SA 4.0 licence](https://huggingface.co/datasets/li2017dailydialog/daily_dialog).
- [Data mirror](https://huggingface.co/datasets/roskoN/dailydialog), pinned revision
  `5214b2a66405abf87fd229e5c1007985501ffe3e`. Read zip text directly; never run the
  mirror's dataset loading script. Retain attribution and applicable share-alike
  terms if redistributing derived data. CoSEM is not part of this experiment.

### Stale live-session input after the correction

The user then hit the new gradient-policy guard because their mounted receipt
lacked the corrected policy. Dataset version **3** was verified Ready and contains
the correction. The user renamed the existing notebook to **ReplyCoach DailyDialog
Qwen**; its stable Kaggle ID remains **141311229**, with slug
`replycoach-dailydialog-qwen`. Preserve that notebook and name on future updates.

Notebook version **4** was Quick Saved without running, submitting the explicit
input reference `tanjingminjereme/replycoach-qwen-pairs-20261028-20/3`. Kaggle's
readback confirms the same notebook ID, version 4, private visibility and input
identity, but canonicalizes the input reference without its version suffix.
An already-running interactive session can retain its old mount: update the
attached dataset to version 3 in the live editor and restart the session before
Run All. The policy guard is intentional and must not be removed to use stale
training code. No training or session cancellation was initiated by this fix.
