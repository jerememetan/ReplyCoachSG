# All-candidate supervised expansion — 4 November 2026

User instruction: “how about just accept everything, and go train”. This authorizes all100 fresh candidates and the existing500-candidate pool. The first10 fresh examples retain explicit individual user approval; the other590 are authorized as a batch, not independently human-reviewed or verified gold. Raw sources, first-batch approval and individual/blanket distinctions are preserved in `data/training_expansion/blanket600_v1/approval_ledger.json`. No preference rankings are invented and no DPO/RL is part of this run.

## Frozen training design

Control: unchanged190 original inputs with the reviewed shared-intent targets,380 style rows +180 next-turn context rows =560 rows. Expanded: original190 +100 fresh +500 existing =790 conversations,1580 style rows +646 next-turn context rows =2226 rows. Additional466 context rows are part of the data intervention; the comparison does not isolate just the600 paired targets from their context supervision.

Both runs: same frozen stage1 mT5 base, seed425, q/v rank16 LoRA alpha32/dropout.1, lr3e-4, batch16, speaker labels, sequential generated-STANDARD-conditioned LOCAL inference. Expanded completes8 full epochs =1120 optimizer updates; control receives the same1120 updates (32 original-data epochs). Equal update count is not equal FLOPs, training-item exposure or reference-loss checkpoint-selection opportunities. Patience100 prevents early stopping;200 epoch cap is a safety bound, not200 actual epochs. Final step1120 is primary; step784 and best-by-reference-loss checkpoints are retained secondary evidence, not post-hoc substitutes.

Actual tokenizer lengths are measured across both complete training corpora before optimization. Common source limit is at least192, rounded up32 as needed; common target limit at least48, rounded up8 as needed. Any unexpected source>512 or target>128 stops for inspection; no silent target truncation. Effective limits/counts are recorded in preflight reports. No new architecture or preference objective is added.

## Evaluation and provenance

Final checkpoints generate the same50 original validation and24 duration-probe development conversations. These reused cases remain development evidence, not a newly untouched benchmark. A fresh anonymous paired packet compares both arms on STANDARD adequacy, LOCAL adequacy, meaning preservation and natural Singlish; uncertain is not pass. The assistant authored100 and previously curated the500 pool and knows prior outputs, so hidden labels are not independent human blinding. No human accuracy or significance claims. References are excluded from generation input.

Source/data SHA256 are frozen in `blanket600_v1_recipe.json`; raw sources and existing reviewed examples remain unchanged. The training manifest checks unique IDs, complete paired targets and normalized protected-message nonoverlap against original training, validation and both prior diagnostic sets. No held-out test file is read. No automatic default replacement. Await accounting, full output verification and reply review before any quality conclusion.

## Completed run and outcome

Control job352816 completed successfully in3m29s; expanded job352817 completed successfully in3m56s, each on one RTX3090 using the two authorized accounts. Both completed1120 updates. The expanded arm completed all8 full epochs of2226 rows (790 conversations plus eligible context pairs); control completed32 epochs of560 rows. Actual maximum source length161 tokens and target length32 fit unchanged192/48 limits, so no training replies were truncated or source/target settings widened. Both train1,769,472 q/v LoRA parameters; peak allocated CUDA memory was5.963GB control and6.861GB expanded. Final, step784 and best-by-reference-loss weights are retained.

Reference loss is not the quality decision: best aggregate reference loss was2.5419 control versus2.5483 expanded. Final STANDARD reference loss was2.2267 versus2.2046; LOCAL loss is oracle-conditioned and must not be presented as independent generation quality. Final matched-step outputs remain primary as frozen before training.

All148 whole-pair outputs on the same50+24 development cases have complete four-axis ratings, frozen before opening arm labels.21 exact unchanged pairs reuse prior assistant evaluation judgments with explicit provenance;127 were newly inspected. These remain74 reused conversations, not148 independent conversations or new human labels. Within-case checks found zero inconsistent single-draft adequacy/style ratings for identical text. The reviewer authored/curated the data and knows prior outputs; hidden labels are not independent blinding.

| Criterion | Control /50 validation | Expanded /50 validation | Control /24 diagnostics | Expanded /24 diagnostics |
|---|---:|---:|---:|---:|
| STANDARD adequacy |36|36|12|9|
| LOCAL adequacy |36|35|12|10|
| Paired meaning |46|48|23|23|
| Recognizable natural Singlish |38|25|11|17|
| All four |28|16|3|7|

Uncertain is not pass. The expanded model has19 exact STANDARD/LOCAL copies among50 validation pairs, versus3 for control (case/punctuation normalized for this copy diagnostic). Identical replies can preserve meaning while failing recognizable Singlish. That helps describe the observed style-control regression; it does not prove whether the cause is target wording, the new646-row context supervision mix, data proportions or another factor. The data intervention changes both600 paired conversations and466 additional context rows; causes were not individually isolated.

**Training is complete; keep the configured default unchanged.** More accepted data did not establish an overall quality upgrade in this one-seed comparison. Validation STANDARD adequacy is unchanged, while recognizable Singlish and all-four counts decline. Some selected diagnostics improve in all-four despite lower STANDARD adequacy, so do not summarize the results as uniformly better or uniformly worse on every measure. These are exploratory assistant judgments, not human accuracy or final-test performance. No extra preference/RL job was started because accepting all targets supplies no chosen/rejected human preference pairs.

Actual examples show mixed behavior. Expanded correctly offers to bring the user's charger and answers the Bay14 question; control reverses the charger borrowing roles and names a railway station. Expanded's LOCAL often copies neutral English, including availability/refusal and acknowledgement replies. Both arms still accept the impossible two-o'clock exam meeting or after-closing lens pickup. The retained outputs and individual reasons are available for inspection; no model was promoted based on a token-loss decrease.

The scope of human feedback is preserved:10 fresh pairs were approved after the user reviewed the first batch, while590 other pairs received blanket permission to enter training. This is not600 individually human-verified examples,600 human preference comparisons or a trained reward function. Original candidate CSVs/preparation manifests remain historical review records; the newer authorization ledger records the user's instruction to proceed without remaining per-item reviews.

## Retained evidence

`blanket600_v1_results.json` consolidates checked counts, curves, source hashes, exact-copy diagnostics and quality summaries. Local raw reports, final/step784 output sets, anonymous packet/key, ratings, consistency audit and SLURM logs are in `artifacts/blanket600_v1/`. Group checkpoints: `~/ReplyCoachSG/experiments/blanket600_v1/artifacts/control_s425/{best,final,step_784}`. Personal expanded checkpoints: `~/ReplyCoachSG_blanket600_v1/artifacts/expanded_s425/{best,final,step_784}`. The8-epoch expanded training corpus is frozen at `data/training_expansion/blanket600_v1/expanded.jsonl`.

Recipe/data/source and retrieved-archive hashes were verified. The paired review completeness/four-axis contract test passed. No trainer implementation changes were needed for this experiment; the existing checkpoint/trainer checks remain applicable. Python compilation and shell syntax passed. Both queues and all retained weight files are checked after completion. No held-out test file was opened, no DPO labels were fabricated and no default checkpoint setting was changed.
