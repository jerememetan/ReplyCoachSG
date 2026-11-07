# Paired synthetic supervision pilot — 60 authored candidates

This is a separate data ablation after the DPO experiment did not establish a
reply-quality gain. It changes training examples, not model architecture or DPO
weight. Existing supervised baseline and previous adapters remain unchanged.

## Data fixed before new model results

Source candidates: `data/synthetic/paired60_v1_candidates.jsonl`, 60 fictional
assistant-authored conversations, ten per group: roles/requests, direct answers,
changed plans/constraints, refusals/counteroffers, context references, and missing
information/new details. Each has STANDARD and LOCAL replies with the same
intended meaning. The last group includes legitimate new status, location,
deadline and approval details, not an automatic clarification-only policy.

No real private messages or locked test examples were supplied. Generation
uses the assistant, with exact model identifier unavailable. Source candidate
review flags remain false; a separate local Gemma run reviews pairs before a
new approved file can be produced. This is different-model review, not human
Singlish validation or proof of independence from shared LLM biases.

TF-IDF unigram/bigram similarity on context+message against frozen train and
validation inputs had maximum cosine **0.319**, with zero cases at the 0.35
manual-inspection threshold. Exact normalized overlaps are also blocked by the
loader. This is not a proof against semantic/paraphrase leakage. The existing
validation set has been reused in development; it is not an untouched benchmark.
No claim of individually human-reviewed synthetic items is made.

Paired review job: **349716**, group account. Prompt/version and every raw
verdict are preserved in `artifacts/paired60_v1_review.json`. Only acceptable
verdicts with no issues can be proposed for training, followed by assistant
inspection. Rejects/uncertainty are not silently rewritten or auto-promoted.

## Fixed training comparison

A new isolated snapshot runs control versus added approved synthetic data for
seeds 425/1/2, same stage-1 base, rank-16 LoRA, lr 3e-4, batch 16, speaker labels
and authored context pairs. No DPO or unlikelihood in either arm. Both use an
8-epoch cap, patience 2 and an optimizer-update cap of **280** (the previous
control's eight × 35 updates). Adding data must not simply buy extra updates.

An augmented epoch contains more batches, so it may end partway through the
last epoch at this cap. That final partial epoch is validated and checkpointed.
Early stopping can end a run before the cap; report actual updates, not just
requested budgets. Equal update counts do not imply equal token/FLOP counts or
equal per-example exposure. Validation checkpoint opportunities occur at epoch
boundaries and therefore are not exactly equal between arms; disclose this
limitation. Existing baseline results are a reproducibility reference, not a
substitute for the fresh matched controls.

Primary qualitative seed 425 is fixed before inspecting new outputs. Compare
all seed loss curves, copied/generic answers and role/negation/question failures.
If other seeds are inspected qualitatively, mark them exploratory. Any assistant
judgments are not human gold or final persona evaluation. Never select a winner
from token loss alone or tune repeatedly on this validation set until a win.

## Status

Paired review completed: Gemma returned acceptable/no-issues for all **60/60**.
The author audit nevertheless held out `refusals_counteroffers-03`: the model's
own explanation calls its LOCAL wording awkward, despite an approval verdict.
No silent rewrite was made. **59** examples are in the separate reviewed file,
`data/synthetic/paired60_v1_reviewed.jsonl` (118 style-specific reply rows).
Five groups have ten items; refusals has nine. Original candidate flags remain
false; original review output remains unchanged.

Reviewed source SHA256:
`f03f327dbb5a169e6d3d3cf45ce3456f53fb23f8887d7795087022554fdc1021`.
Review/model digest, settings, all raw reasons and per-item author audit are saved
in ignored `artifacts/paired60_v1_review.json` and
`artifacts/paired60_v1_author_audit.json`. Both reviewers are models, not humans.
The legacy `facts_supported` training flag here means compatible fictional
context or an allowed new selectable-draft detail, not real-world verification.

A new six-run control/augmentation batch is submitted from its own frozen
source/data snapshot. The source 12 starter examples remain unreviewed and are
not mixed in. No existing checkpoint is overwritten.

Code checks: **42** trainer/synthetic-loader/pair-review tests passed, including a mid-epoch update-cap test, invalid-cap rejection, metadata isolation and inconsistent-review rejection. Shell and Python syntax checks passed.

Training job **349763** submitted under `CS425G9`, isolated work directory `~/ReplyCoachSG/experiments/synthetic_paired60_v1/`.


## Completed results

Paired review job **349716** completed in **3m 10s** on `voodoo`. Training job
**349763** completed in **5m 50s** on `lava`, group account `CS425G9`, exit 0.
One RTX 3090 with 24 GB VRAM, 32 GB system RAM and four CPU cores was allocated
for training. All six runs reached exactly **280 optimizer updates** and produced
both styles for all 50 validation conversations. No jobs remain running.

| Seed | Control best validation loss | Added 59 conversations | Augmented minus control |
| --- | --- | --- | --- |
| 425 | 2.4990 | 2.5332 | +0.0342 |
| 1 | 2.5075 | 2.5283 | +0.0208 |
| 2 | 2.5139 | 2.5309 | +0.0170 |
| Mean | 2.5068 | 2.5308 | +0.0240 |

Higher loss is worse. The added-data arm was worse in all three seeds. Mean
STANDARD loss delta was +0.0329; LOCAL +0.0151. This is reference-token loss on
reused validation data, not an accuracy or quality percentage.

The new controls reproduced **all three previous control losses and every
validation reply exactly**, despite the added update-cap reporting. This is a
useful reproducibility check that the control training behavior was preserved.

### Primary reply review

| Outcome, seed 425 | Count |
| --- | --- |
| Control preferred | 19 |
| Augmented preferred | 16 |
| Tie | 25 |
| Both bad | 10 |
| Exact identical | 30 |

Full per-pair context and rationales are in
[the validation review](synthetic_paired60_validation_review.md). These are
anonymized assistant judgments, not fully blind or independent human ratings:
the reviewer authored the data and had already seen control replies in the DPO
study. Neither this review nor the loss result establishes the final test result.

The augmented model fixed some useful cases, including explaining missed class
to the absent peer and offering the requested charger handoff instead of
reversing who uses the charger. It also regressed elsewhere: a stall-opening
question became a dinner-availability response; a meal-location question became
an already-known hunger statement; a meeting-time confirmation became checking
with the peer at that time. Gains and regressions are mixed, not a reliable upgrade.

### Budget, copying and overfitting checks

Control used **560** rows (380 authored style replies + 180 authored context
pairs), augmented **678** rows (plus 118 synthetic style replies). All runs used
280 updates. Controls completed eight epochs; augmentation ended in a partial
seventh epoch. Thus augmentation sees each authored conversation fewer times,
and checkpoint opportunities occur at different update counts. This experiment
asks whether these data help **at a fixed update budget**, not whether they can
help with extra training or a different mixture. Equal updates are not equal
FLOPs/tokens because sequence lengths and final partial batches differ.

Best checkpoints: control steps 280/280/245 for seeds 425/1/2; augmentation steps
258/280/280. Augmented seed 425 reached loss 2.5332 at step 258, then rose to
2.5403 at step 280. Other augmented losses were still improving. This single
upturn does not prove overfitting; short-run curves do not rule out memorization
or poor semantic transfer either. No early stopping triggered before the cap.

Exact normalized context-turn copies across the repeated three-seed 300-output
cells increased from **3 control to 13 augmented**; exact incoming-message echoes
were **3 in each arm**. Copying known facts can be appropriate, so these are
heuristics rather than correctness measures. A concrete failure repeats the
user’s printer-instruction question after the peer already said they do not know.
All runs had zero blank outputs. Peak allocated memory was about **5.5–5.7 GB**.

### Scope and likely follow-up hypothesis

This pilot is mostly standard-English incoming dialogue, although its outputs
have paired STANDARD/LOCAL styles. The real development chats have much more
shorthand, Singlish and code-switching. That mismatch is a plausible hypothesis,
not an established cause of the result. The 59 situations also introduce
vocabulary outside the small original task set. The dataset and fixed compute
budget are limited; do not conclude that all synthetic supervision fails.

The old plan’s 190/570 added-conversation volume arms were **not** run here.
This 59-conversation pilot is not presented as completing that full volume study.
A sensible next distinct ablation would hold these scenarios/replies fixed and
vary incoming chat register, checking meaning again before training. Do not
simply increase volume or tune on the same validation replies until a win appears.

## Decision and artifacts

**Retain the existing supervised baseline.** No default checkpoint, provider or
deployment changed. Keep these six adapters as experimental artifacts, with a
negative result for the tested data recipe at this budget. No architecture,
DPO-weight or synthetic-volume ablation is silently bundled into this conclusion.

- Frozen recipe/code/data hashes: `experiments/synthetic_paired60_v1.json`.
- Authored candidates: `data/synthetic/paired60_v1_candidates.jsonl`.
- Reviewed training file: `data/synthetic/paired60_v1_reviewed.jsonl` (59 items).
- Ignored paired-model review and author audit: `artifacts/paired60_v1_{review,author_audit}.json`.
- Ignored six run reports, analysis, curves CSV, anonymous packet/key and ratings:
  `artifacts/synthetic_paired60_v1/`.
- Cluster snapshot/checkpoints:
  `CS425G9:~/ReplyCoachSG/experiments/synthetic_paired60_v1/artifacts/{control,synthetic}_s{425,1,2}/best`.
- Training runner: `scripts/v5_synthetic_compare.sbatch`.
- Analysis/review helpers: `scripts/analyze_synthetic_comparison.py`,
  `scripts/prepare_generation_review.py`, `scripts/score_generation_review.py`.

The locked test file was not copied or read. Original train/validation files,
candidate/review provenance and existing model checkpoints remain unchanged.
42 targeted trainer/loader/reviewer tests passed, including partial-epoch budget
stopping, malformed budgets and contradictory review verdicts. Post-run checks
verified all six step counts, 50 paired validation inputs, current settings and
exact control reproducibility. Python/shell syntax and diff checks passed.
