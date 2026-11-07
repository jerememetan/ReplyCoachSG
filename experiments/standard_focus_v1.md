# STANDARD adequacy: three parallel experiments

User authorized parallel experimentation on 4 November 2026; three concurrent experiments are not a three-run total budget. Initial wave uses one common fresh control and three treatments. Two available GPUs imply at most two simultaneous GPU jobs; SLURM queues the third. No production replacement or final-test access.

## Frozen design

All runs start from the same speaker-labelled stage-1 mT5 base, seed425, lr3e-4, batch16, source192/target48, rank16/alpha32/dropout.1, 784 optimizer updates. Primary is final step784; best-by-reference-loss checkpoints remain secondary and must not replace the primary post hoc. All retain STANDARD-to-LOCAL rewriting. Context-pair auxiliary rows stay the original180. Original190 inputs use the reviewed paired targets from shared_intent_v1.

- Experiment data: fresh control (560 rows) versus +80 targeted conversations (720 rows), attention q/v adapters.
- Experiment capacity: same targeted corpus and q/k/v/o/wi_0/wi_1/wo adapters. This adds parameters/compute; it is a capacity experiment, not matched FLOPs.
- Experiment STANDARD emphasis: same targeted corpus and q/v adapters; each STANDARD row occurs twice and each LOCAL rewrite once (990 total rows). Fixed updates mean changed exposure, not extra optimization steps; LOCAL exposure decreases proportionally.

The data experiment establishes whether added contrasts help. Capacity and STANDARD-emphasis treatments must also be compared against targeted-data-only to isolate their extra changes. All interventions are hypotheses, not established improvements.

## Data and evaluation

80 assistant-authored contrasts, ten families with four lexical settings and two alternatives each: lending/borrowing; sending/receiving; who knows how; accepting/refusing; temporal constraints; finished/unfinished work; known/unknown locations; who updates whom; returns/ownership; dietary constraints. Templates are intentionally correlated and are not80 independent domains. Data are assistant-authored/audited, not human gold. No Gemma generation or autonomous critic labels are used for these additions.

Frozen additional24 diagnostic contexts are authored before training and never passed to the trainer. They are development diagnostics, not a replacement held-out test or independent human evidence. Plausible selectable new details, valid refusals and alternatives remain allowed; exact reference matching is not required. Review STANDARD adequacy separately, then LOCAL adequacy, meaning preservation and recognizable natural Singlish. Judge anonymized output packets before opening arm keys; the reviewer authored examples and knows prior behavior, so this is not independent blinding.

Compare all four arms on the same50 existing validation conversations and24 fresh diagnostic contexts. Keep reference replies/diagnostic requirements out of model generation prompts. Report failure examples and uncertainty; do not treat keyword matching as adequacy. An apparent one-seed gain requires fresh-seed replication before model promotion. Reusing validation and authored diagnostics limits conclusions. No iterative tuning against the same diagnostic labels.

## Execution

Snapshot hashes are recorded in `experiments/standard_focus_v1_recipe.json`. Jobs and completion results will be appended after observed accounting and artifact verification. No result is assumed at submission.

## Initial wave completed

Group job352584 completed control plus targeted-data in4m43s; group job352587 completed capacity in3m10s. Personal job352599 completed STANDARD emphasis in2m37s. All four reached784 updates and saved best/final adapters,50 validation pairs and24 diagnostic pairs. Capacity has6,782,976 trainable parameters versus1,769,472 in the q/v runs. CUDA peak allocation is recorded in the raw reports. All successful runs used RTX3090 GPUs. Local targeted trainer/provider checks:51 passed; review-completeness/split-isolation test:1 passed. Shell syntax and normalized protected-message checks passed.

The scheduler allows one active/one-GPU group job and two submitted group jobs. An initial third submission and a two-GPU request were rejected. The second account was provisioned with an exact copy of the same Python environment and base checkpoint, using a temporary transfer ACL restricted to that account; original repository permissions were not broadened and the temporary transfer directory was removed. Personal attempt352589 failed on an unsupported GTX1080Ti before training updates; it was retried with an explicit3090 constraint. Group pending job352585 was canceled before execution. These are infrastructure events, not failed model-training outcomes. See the launch manifest.

Fresh control best loss exactly matched the previous rewrite pilot;49/50 generated pairs matched. Do not claim bitwise reproduction. Current adequacy ratings are fresh and differ from earlier assistant ratings; compare arms within this review rather than claiming the baseline improved over historical counts.

All296 pairs were rated on four explicit axes before opening the arm key. References and model labels were hidden in the review packet. Six noted naturalness uncertainties were aligned with their recorded repetition/word-order concerns before unblinding; the audit is preserved. No ratings were edited to favor a model after unblinding.

| Arm | STANDARD pass /50 | LOCAL adequacy /50 | Meaning /50 | Singlish /50 | All four /50 | STANDARD /24 diagnostic | All four /24 diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control |38|35|43|39|26|9|4|
| Targeted data |35|32|46|34|22|12|6|
| Broader adapters |37|32|45|39|28|12|10|
| STANDARD emphasis |33|30|43|33|21|12|6|

Uncertain is not pass. Diagnostic contexts deliberately stress errors and are not representative of ordinary traffic. All arms fail the egg-allergy, injured-wrist lifting and after-closing pickup cases; novel keys and coat-return roles also remain difficult. The targeted-data-only arm correctly accepts receiving recorded audio, whereas broader adapters handle the unfamiliar locker by asking another person. Broader adapters still reverse the audio direction, and the targeted-data-only arm asks the uninformed person for locker instructions. All treatments repair the charger-lending example, but that alone does not establish general role understanding.

Policy ambiguity remains: an unmentioned own spare card can be a valid alternative to a peer's offered card, so it was not automatically rejected; time between unfinished and newly finished work can make contextual completion uncertain even when cross-style contradiction is definite. Frozen diagnostic adequacy notes are assistant guidance, not gold labels. The raw examples and uncertainty remain available; no post-hoc relabeling of training examples occurred.

## Capacity replication submitted

The broader-adapter candidate has a mixed but promising paired-quality signal, not an established STANDARD improvement. Repeat control versus capacity with seeds1/2, unchanged data/settings, final step784 primary, no new prompts or target edits. Seed425 selected the candidate, so these remain development comparisons. The same50/24 contexts are reused; additional seeds are not additional independent conversations. Recipe: `standard_focus_v1_replication_recipe.json`. Final decision awaits observed completion and fresh anonymous ratings.

## Replication complete and final decision

Jobs352606 (group, seed1) and352607 (personal, seed2) completed successfully in5m11s and5m15s on RTX3090s. Each trained control plus the targeted-data/broader-adapter candidate. Eight successful training runs total across both waves; each completed784 updates, retained best/final adapters and generated50 validation plus24 diagnostic pairs. All reports, expected seeds, training row counts, original-only context supervision and artifact hashes were checked. Both account queues were checked after completion.

The three-seed replication tests the **combined targeted-data plus broader-adapter intervention**, not adapter capacity in isolation. Only seed425 includes the targeted-data-only and STANDARD-emphasis ablations.

| Seed | Control STANDARD /50 | Candidate STANDARD /50 | Control all-four /50 | Candidate all-four /50 | Control all-four /24 diagnostic | Candidate all-four /24 diagnostic |
|---|---:|---:|---:|---:|---:|---:|
|425|38|37|26|28|4|10|
|1|38|35|30|27|10|12|
|2|37|38|28|31|7|13|

Mean pass counts across the same three runs:

| Criterion | Control /50 validation | Candidate /50 validation | Control /24 diagnostic | Candidate /24 diagnostic |
|---|---:|---:|---:|---:|
| STANDARD adequacy |37.7|36.7|10.7|14.0|
| LOCAL adequacy |36.3|34.0|10.3|13.0|
| Paired meaning |46.3|43.7|22.7|19.0|
| Natural recognizable Singlish |40.3|43.3|18.7|21.3|
| All four |28.0|28.7|7.0|11.7|

**Keep the configured default unchanged. No general STANDARD-quality upgrade is established.** The combined candidate improves selected diagnostic adequacy/all-four counts across seeds, but validation STANDARD is slightly worse on average, validation all-four changes are mixed, and meaning consistency declines on average on validation and in every seed on diagnostics. Better Singlish and targeted-diagnostic performance do not compensate automatically for lost meaning preservation. The earlier sequential-rewrite finding remains useful; these results do not invalidate it.

The targeted-data-only and STANDARD-emphasis initial runs did not improve general validation quality. Do not continue sweeping variants against the same24 diagnostic contexts until a winner appears. These negative/mixed results narrow the tested hypotheses; they do not prove that all targeted supervision, broader adapters or task separation fail. Best-by-loss checkpoints remain retained secondary artifacts, not post-hoc substitutes for the frozen primary.

All592 whole pairs across the three packets have four-axis ratings. Replication packets reuse26 and24 exact unchanged case-and-draft pairs from the frozen seed425 review, with explicit provenance; the other246 replication pairs were newly inspected. These are not independent ratings or new conversations. Within-packet and cross-seed checks found zero inconsistent single-axis judgments for identical draft text on the same case. This establishes internal consistency only, not human validity. Review completeness/seed handling tests now pass for both four-arm and two-arm packets (2 tests), alongside the51 training/provider checks.

The same assistant authored training examples, wrote diagnostic expectations, and rated outputs while knowing prior behavior. Model IDs were hidden before ratings but may be inferred. Diagnostics are24 correlated authored development scenarios, not a representative or independently validated benchmark. Seeds repeat the same50/24 conversations; do not pool them into222 independent conversations. Completion-timing and plausible-spare-card ambiguities were retained rather than forcing agreement with authored diagnostic notes. No human accuracy, significance, deployment improvement or final-test score is claimed. No RL was run, no held-out test was opened, and no checkpoint was promoted.

Detailed results/hashes: `standard_focus_v1_results.json`; three-seed comparisons: `standard_focus_v1_replication.json`; frozen recipes and launch/provenance manifests accompany them. Local anonymous packets, ratings, keys, audits, reports and SLURM logs: `artifacts/standard_focus_v1/`. Group checkpoints: `~/ReplyCoachSG/experiments/standard_focus_v1/artifacts/`; personal checkpoints: `~/ReplyCoachSG_focus_v1/artifacts/`. The second account's copied environment/base remain available for reproducibility; no original account permissions were broadened.
