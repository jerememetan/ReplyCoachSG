# Supervised-only versus supervised + DPO — reviewed 73 preferences

Job **349499**, `CS425G9`, node `lava`, one RTX 3090 (24 GB VRAM), 32 GB
system RAM, four CPU cores. Work directory:
`~/ReplyCoachSG/experiments/dpo_review73_v1`. Existing baseline checkpoints and
cluster checkout changes are preserved. This experiment runs from its own
source/data snapshot with verified SHA256 hashes.

## Fixed comparison

- Same stage-1 speaker-labelled base: `artifacts/mt5_base_after_spans_spk/best`.
  Each arm starts a fresh rank-16 LoRA adapter; this does not resume/overwrite
  the existing selected stage-2 baseline.
- Same 190 authored training conversations, 380 style replies and training-only
  context pairs. No synthetic examples or echo unlikelihood in either arm.
- Seeds **425, 1, 2**. Both arms use learning rate **3e-4**, batch **16**,
  maximum **8 epochs**, patience **2**, LoRA dropout **0.1** and alpha **32**.
- DPO arm adds the **73 assistant-reviewed preferences**, weight **0.1**, beta
  **0.1**, against fixed stage-1 reference log-probabilities. The supervised
  objective remains active. Preferences are sampled alongside each supervised
  batch and therefore repeat across training steps.
- Best checkpoint within each run is selected by mean STANDARD/LOCAL token loss
  on the same **50 validation conversations**. It is not selected on test data.
- 73 preferences cover **61** training conversations; they are not independent
  human gold. User review established six policy examples. Earlier blanket
  unknown-fact rejection is not used.

Recipe/hashes: `experiments/dpo_review73_v1.json`. Job entrypoint:
`scripts/v5_dpo_compare.sbatch`. Before submission, **49 targeted tests passed**.

## Evaluation plan

Compare paired losses and learning curves across all three seeds. Inspect token
loss and output quality separately; training DPO agreement is not held-out
semantic accuracy. Measure basic repetition, exact message/context copies and
same-text-in-both-styles rates as diagnostics, not accuracy.

Use seed **425**, fixed before inspecting DPO replies, for an anonymized assistant
comparison on all 50 validation conversations × two styles. Identical replies
are recorded separately, not assumed good. Hide arm IDs while choosing between
different replies, then unblind after recording judgments. This is assistant
inspection with shared development context, not independent human evaluation.
Cross-style meaning and contradictions need direct inspection; personal details
without prior evidence are allowed as selectable drafts.

Do not pick a winner from token loss alone or promote these validation results
to a final test score. If the DPO arm fails to improve reply quality, retain the
supervised baseline and report the negative/inconclusive result. Do not tune
repeatedly until this reused validation set prefers DPO.

## Status

**Completed successfully** (job 349499, exit 0) in **8m 05s**. All six runs
produced all 50 validation conversations in both styles. The locked test file was
not included in the snapshot and was not opened. Reports were retrieved locally
under ignored `artifacts/dpo_review73_v1/`.

The three-seed qualitative inspection below is additional exploratory work;
only seed 425 was fixed as the primary qualitative review in advance.


## Results

| Seed | Supervised-only best loss | Supervised + DPO best loss | DPO minus control |
| --- | --- | --- | --- |
| 425 | 2.4990 | 2.4893 | -0.0097 |
| 1 | 2.5075 | 2.4981 | -0.0094 |
| 2 | 2.5139 | 2.4980 | -0.0159 |
| Mean | 2.5068 | 2.4951 | -0.0117 |

Lower token loss is a small consistent improvement here, not a reply-quality
accuracy score. Mean STANDARD loss delta is -0.0087; LOCAL is -0.0147.

| Seed | Supervised preferred | DPO preferred | Tie | Both bad | Identical |
| --- | --- | --- | --- | --- | --- |
| 425 | 11 | 7 | 20 | 11 | 51 |
| 1 | 8 | 10 | 18 | 15 | 49 |
| 2 | 13 | 7 | 29 | 10 | 41 |
| Descriptive total | 32 | 24 | 67 | 36 | 141 |

These are assistant judgments, not independent human gold or final evaluation
persona ratings. The same conversations appear across seeds, so the 300 cells
must not be treated as 300 independent conversations. The 36 both-bad ratings
apply only to differing reply pairs; identical outputs were not assumed good,
so this is not an overall bad-reply rate.

Complete per-case context, both replies and rationale are in
[the validation review](dpo_review73_validation_review.md). Raw blinded packets,
keys and ratings were saved separately, with ratings fixed before unblinding.
Some preferences are debatable; no statistical superiority claim is made.

### Concrete mixed behavior

- Seed 425, ITEM-A-046: proposed time is 2 pm. Control says “2pm sounds good”;
  DPO says it will arrive at 3 pm without framing a counterproposal. Control
  preferred. Seed 2 reverses this pattern: DPO says 2 pm and control says by 3 pm.
- Seed 1, ITEM-C-018: DPO responds with merging slides, while the control says it
  will forget them. DPO preferred in both styles for that seed.
- Seed 2, ITEM-C-038: control answers a length-check request with a cost answer;
  DPO offers to check the passage. DPO preferred.
- ITEM-D-038: the user is already outside Mac, and the peer asks them to wait.
  Both seed-425 LOCAL replies instead say the user is arriving. DPO fixes this
  for seed 1, but control is better for seed 2. Role problems remain inconsistent.

## Overfitting and resource checks

All runs completed the eight-epoch cap; early stopping did not trigger. Best
epoch was eight for five runs, seven for control seed 2. That run’s validation
loss rose only 0.0001 at epoch eight while training loss fell. There is no clear
sustained token-loss divergence within this short run, but that does **not**
rule out memorization, judge-bias learning or poor semantic generalisation.
Most best checkpoints are at the cap, so longer-run stability was not tested.

All arms start at the same stage-1 validation loss of 3.0005. End supervised
training loss is 2.3157–2.3409 for control and 2.3110–2.3194 for DPO. Training
and validation losses use different examples/modes and are not directly an
independent accuracy measure. DPO’s reported 0.707–0.730 training agreement is
relative movement against reference log-probabilities, not held-out preference
accuracy or a real-world quality percentage.

Peak allocated GPU memory: control **5.61–5.68 GB**, DPO **9.83–9.94 GB**.
Epoch training/validation time per run: control **36.8–37.4s**, DPO **78.6–79.0s**;
model loading and output generation are additional, explaining job wall time.
Each model has **584,170,752** parameters, of which **1,769,472** LoRA parameters
are trained. Runtime packages: torch **2.14.1**, transformers **5.19.0**, peft
**0.21.2**, Python module **3.13.1**.

Basic diagnostics show zero blank replies. Each run has one exact message echo
in LOCAL; exact copies do not capture all wrong-role paraphrases. Normalized
same-text-in-both-styles counts vary by seed (control 8/1/7; DPO 7/9/8 out of 50).
These are diagnostics, not style or semantic-equivalence scores. Reply quality
and style agreement need direct inspection.

## Decision

**Do not replace the existing supervised baseline with these DPO adapters.**
This small reviewed-preference recipe slightly improves token loss but does not
establish a consistent reply-quality gain across seeds. Archive the experiment
as inconclusive for quality, with mixed examples. Existing selected checkpoint
`artifacts/spk1_ul025_s425/best` remains unchanged; the short-run controls are
experimental comparators, not a newly selected production baseline.

All six adapters are retained on the group account under
`~/ReplyCoachSG/experiments/dpo_review73_v1/artifacts/{sft,dpo}_s{425,1,2}/best`.
No deployment or default checkpoint change was made. No jobs remain from this
comparison and no additional training is automatically scheduled.

Next research priority is better paired task supervision for the repeated role,
negation and question-answering failures, evaluated as a separate data ablation.
Do not keep tuning DPO against this reused validation set until a win appears.
Main blinded persona evaluation and the V4/V5/hybrid comparison remain separate
project tasks; this assistant review does not replace them.

## Reproduction and artifacts

- Recipe, code/data hashes: `experiments/dpo_review73_v1.json`.
- Snapshot runner: `scripts/v5_dpo_compare.sbatch` (submit from the frozen
  experiment snapshot, setting PROJECT_ROOT to the original group repository).
- Analysis: `python scripts/analyze_dpo_comparison.py artifacts/dpo_review73_v1 --output <fresh-summary.json>`.
- Local ignored raw reports, summary, learning-curves CSV, blinded packets and
  ratings: `artifacts/dpo_review73_v1/`.
- Full qualitative review: `experiments/dpo_review73_validation_review.md`.
- The updated source was tested before submission: **49 targeted tests passed**.
  Post-run analysis verified matched settings, all 50 validation IDs and identical
  contexts/reference replies within every seed pair. No test-set content was read.
