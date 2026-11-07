# Fixed-trajectory duration probe — 4 November 2026

User authorized testing training duration after STANDARD-quality experiments remained mixed. Two runs: current experimental rewrite control (original190 reviewed target pairs, q/v adapters) and the combined targeted80-data plus broader-adapter candidate. Each starts at the same frozen stage1 base, seed425, lr3e-4, batch16, rank16/alpha32/dropout.1, speaker labels, unchanged180 context auxiliary rows, source192/target48, patience100,784 updates. Preserve distinct arms and interpret within-arm duration first; between arms changes both data and adapter scope.

Save exact checkpoints at160/320/480/784 without inserting evaluation, generation or extra optimization into the trajectory. Checkpoint capture was tested for bitwise final adapter equality and identical final generations on a tiny model. Source hashes, data hashes and actual token limits are checked on cluster. Primary is inspection of all four predeclared steps, not retrospective substitution of a best-by-token-loss checkpoint. Two accounts use the same Python environment/base copies on RTX3090 GPUs.

24 new assistant-authored development scenarios were frozen before either submission. They mix ordinary acknowledgements with role, offer/request, availability, dietary/physical constraints, completion, explicit-location and knowledge cases. No normalized protected-message matches against prior train/validation or the earlier24 diagnostics; this is not a semantic-independence guarantee. Cases are not independent human gold, representative traffic or the locked final test. No fresh probe examples/guidance enter training, reference-loss checkpoint selection or generation prompts. Fresh data are loaded only after the full training trajectory is complete. Four-axis assistant inspection: STANDARD adequacy, LOCAL adequacy, paired meaning, recognizably natural Singlish. Valid refusals, counterproposals and plausible selectable details remain allowed; uncertainty is retained.

Review all eight checkpoint variants in anonymized per-case order before opening the arm/step key. Reviewer authored scenarios and knows prior behavior, so hidden IDs are not independent human blinding. Primary STANDARD adequacy must be interpreted alongside all-criteria outcomes; no style gain excuses incoherent replies or meaning reversals. The probe is used to explore checkpoint duration, so picking a winner here is development selection, not an unbiased final performance estimate. No default promotion or final-test access is automatic.

Original50 validation outputs and STANDARD reference loss are retained at each step as secondary diagnostics. Generating them is not a new quality review or accuracy claim. Compare reloaded step784 with the trainer's final outputs, and compare final reports against the preceding frozen runs to check trajectory consistency. All earlier results remain intact. No RL is part of this run. Report source/result hashes and observed accounting before claiming completion.

## Completed results

Both trajectories reached784 optimizer updates and saved all four step adapters plus best/final weights. Group control job352662 completed training and three inference artifacts but failed its final text-reproduction assertion. Personal broader-adapter job352663 completed successfully in4m01s. Control recovery job352670 completed inference-only recovery in19s; no training updates or model/decoder/prompt changes were made. Original failed logs and frozen source remain preserved.

The control step784 and final adapter tensors match exactly, as do input, adapter and tokenizer configurations. Reloaded secondary generations matched48/50 of the trainer's in-memory final pairs. The two differences are preserved in `artifacts/control_reload_audit.json`; the cause of the generation differences is not established. They are not evidence that weights changed. The broader arm passed the original all50 text-reproduction check. All primary comparisons use outputs generated from reloaded step checkpoints. Reproduction limitations mean small count differences must not be interpreted as precise deterministic improvements.

Compared with preceding frozen runs, the new control in-memory final generations matched48/50 pairs and final STANDARD reference loss differed by0.0001; the broader arm matched50/50 and the displayed loss exactly. Minor control loss differences were already present before the first step160 snapshot. Do not claim bitwise real-GPU reproduction from the tiny-model capture test.

All192 fresh whole-pair variants received explicit ratings on all four axes before opening the arm/step mapping. Exact duplicate case/pair outputs share a recorded judgment with provenance. Identical single-draft text within the same case had zero inconsistent adequacy/style ratings. Uncertain is not pass. These are assistant judgments on the same24 authored development scenarios, not192 independent cases or human accuracy.

| Arm | Step | STANDARD reference loss on original50 | STANDARD adequate /24 | LOCAL adequate /24 | Same meaning /24 | Natural Singlish /24 | All four /24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control |160|2.2612|9|9|20|16|5|
| Control |320|2.1605|11|10|23|15|6|
| Control |480|2.1453|11|10|24|13|6|
| Control |784|2.1724|12|11|24|13|5|
| Targeted data + broader adapters |160|2.2133|10|9|21|17|8|
| Targeted data + broader adapters |320|2.1923|11|11|17|17|6|
| Targeted data + broader adapters |480|2.2770|12|12|19|18|9|
| Targeted data + broader adapters |784|2.5770|12|13|19|19|9|

**Earlier stopping did not establish better STANDARD replies on this probe.** Both final variants have12 STANDARD passes, versus best earlier11 for control and12 for the broader arm. Broader step480 ties step784 on all-four (9), while step320 has6 despite its lower original-reference loss. Control steps320/480 have one more all-four pass than its final step, but neither has better STANDARD adequacy. These small mixed differences do not establish a deployment upgrade. No checkpoint was promoted, no additional training was triggered and the configured default remains unchanged.

The original-reference losses confirm late loss deterioration, especially for the broader arm. This did not translate into a demonstrated earlier-checkpoint reply-quality win here. The experiment does not prove that overfitting never affects quality; it separates the observed loss trend from the claimed response improvement.

## Actual error patterns

- All variants fail to give known Bay14 when directly asked which bus bay to queue at. Repeating the south terminal is insufficient to identify the bay.
- Bike-lock requests are commonly answered by lending a bicycle, claiming the peer has their own lock, or saying the user will use the lock. Earlier checkpoints do not resolve that intent/object problem.
- Broader steps480/784 correctly accept receiving the peer's attendance-list copy; the other variants reverse the sending direction.
- STANDARD at broader steps480/784 refuses the incompatible two-o'clock exam-room meeting and proposes tomorrow; LOCAL at480 loses that date/place, while784 preserves the proposal. Earlier checkpoints, including broader320, accept two or produce contradictory timing.
- Control480 identifies the fountain correctly; the other variants change movement/location or give an unclear new plan. Broader784 correctly accepts the cafe refusal and offers another occasion, whereas some earlier replies still press for attendance.
- Many variants still accept the restricted run or after-closing collection. Natural Singlish and matching meanings cannot compensate for these context failures.

Some boundaries remain subjective: a changed requested review week can be a valid new personal plan; an unspecified pickup method may be unclear rather than definitively impossible; new completion since a previous unfinished turn can be uncertain. Raw outputs, axis reasons and uncertainty are retained. No diagnostic guidance was treated as independent human gold.

## Verification and retained artifacts

53 targeted trainer/rewrite/teacher tests passed, including exact-step capture and final-weight/generation invariance on a tiny model, plus1 eight-variant review completeness/axis test. Python compilation and batch-shell syntax passed. Training sources/data and actual tokenizer budgets were checked remotely. All eight step adapters and all24/50 per-step output sets are retained; source/result hashes are in `duration_probe_v1_recipe.json`, `duration_probe_v1_results.json` and the verification/launch manifests. No locked test data was read and no RL was run.

Local reports, packets, key, ratings, reload audit and SLURM logs: `artifacts/duration_probe_v1/`. Group models: `~/ReplyCoachSG/experiments/duration_probe_v1/artifacts/control_s425/`; personal models: `~/ReplyCoachSG_duration_v1/artifacts/capacity_s425/`. The original50 output pairs at each step were retained as secondary evidence; they were not newly quality-rated in this experiment. The fresh24 cases remain development data after this exploration, not a future untouched test.
