# mT5 shared-intention Singlish rewrite pilot

The user requested implementation, with three primary requirements: recognizable natural Singlish, approximately the same meaning across styles, and coherent replies regardless of reference wording. mT5 remains the student/deployment target.

## Implemented path

Train STANDARD conversation-to-reply generation and a separate LOCAL_REWRITE task in the same mT5 adapter. LOCAL_REWRITE receives conversation plus the chosen standard draft. Inference first generates STANDARD and then conditions LOCAL on that exact generated string, never on a reference. No additional runtime model is required. Checkpoint metadata selects the mode in the existing V5 app provider. The production default is unchanged.

Old history may be dropped to respect an input budget, but the chosen standard anchor is never silently truncated; an over-budget rewrite raises an explicit error. Empty standard generation is recorded as an empty failed pair during evaluation without attempting an empty-anchor rewrite. The app continues to reject empty candidates. A shared anchor encourages consistency but does not guarantee it or repair a contextually wrong STANDARD answer.

The optional `--save-final` saves final-step weights and generation outputs in addition to best-by-loss checkpoints. All primary comparisons use the same final update. Rewrite LOCAL token loss is conditioned on the authored reference STANDARD (oracle), so it is not comparable to independent LOCAL loss as a quality score.

## Data and arms

Retain authored standard intentions for 190 original training conversations, with 14 recorded assistant corrections to problematic/awkward targets. A frozen Gemma call rewrites only LOCAL from that fixed standard. No original LOCAL targets, validation or test inputs are supplied to this generator. Generation job 351058; all generated pairs remain unapproved until an assistant audit. No automatic reward-judge approval is used.

Pilot arms: original targets with independent generation; aligned/stronger-Singlish targets with independent generation; the SAME aligned targets with sequential rewrite generation. This separates target changes from conditioning changes. Seed 425 is fixed in advance. All arms use the same mT5 base, rank-16 q/v LoRA, alpha32/dropout.1, lr3e-4, batch16, 784 optimizer updates, patience100, speaker labels, 180 unchanged authored context rows, source limit192 and target limit48. Each has 560 training rows. Different source lengths imply unequal token/FLOP counts despite matched updates.

## Verification and decision

58 tests passed covering existing training/provider behavior, generated-versus-reference anchor isolation, actual speaker roles, truncation protection, empty-generation behavior, final checkpoint metadata and provider scoring on the correct anchored input. No existing user changes or historical outputs were discarded.

Review final-step outputs as whole pairs using the three user requirements. `scripts/prepare_paired_quality_review.py` creates anonymous paired packets with references hidden and arm mapping separate. Assistant judgments remain exploratory and are not independent human gold. Lexical reference loss and particle counts do not decide success. No default replacement or RL run is automatic. Expand to other seeds only if this initial pilot provides a useful signal.


## Data and code ready

Generation job 351058 completed in 5m41s with 188 distinct valid LOCAL candidates and two identical-to-standard failures. The assistant inspected all source/standard inputs and LOCAL drafts, revised 117 LOCAL targets (including repairing those two failures), and retained the 14 recorded standard-anchor corrections. Final corpus is reordered to the unchanged original 190 ID order. Corpus: `data/distillation/shared_intent_v1_reviewed.json`; full before/after ledger: `artifacts/shared_intent_v1/target_audit.json`. Target changes are substantial assistant curation; do not attribute them to raw Gemma output or call them human gold.

Validation passed for complete original input/role/ID coverage, 380 paired target rows and 380 sequential-task rows. The 58 trainer/provider/anchor tests and three paired-review contract tests passed (61 total across the two runs). The paired scorer requires all three criteria separately and does not allow meaning preservation to compensate for failed Singlish style. Frozen training recipe: `experiments/shared_intent_v1_recipe.json`.


Training job **351086** observed RUNNING on `lava`, one RTX 3090 (24 GB). Data hashes, audit flags, 190 input IDs, and complete anchor/target token-budget checks passed. Three arms × seed425, final checkpoint at update784 is primary; no results yet at submission.


## Pilot results

Job 351086 completed successfully in 6m43s. All three arms completed 784 updates and saved best and final weights plus all 50 final-step output pairs. Dataset sizes, actual rewrite anchors and source report hashes verified. Results and hashes: `experiments/shared_intent_v1_results.json`.

After recording all 150 three-axis assistant ratings, the arm key was opened:

| Arm | Context pass | Same meaning pass | Singlish pass | All three pass |
|---|---:|---:|---:|---:|
| Original | 35/50 | 17/50 | 27/50 | 8/50 |
| Aligned targets, independent | 32/50 | 19/50 | 34/50 | 10/50 |
| Aligned targets, sequential rewrite | 32/50 | 42/50 | 37/50 | 23/50 |

Uncertain judgments are not passes. These are subjective, one-seed assistant development judgments, not independent human accuracy. The reviewer authored substantial target corrections, has seen prior model behavior, and could infer identities from outputs despite hidden arm labels. Each arm uses the SAME 50 development conversations, not 150 independent cases.

Interpretation: shared generated-standard conditioning has a promising paired-meaning effect; target changes alone did not resolve meaning divergence. Recognizable local wording also improved in this inspection. Contextual adequacy did not improve clearly and many wrong replies remain. A pair can preserve the same wrong answer; a plain-English copy also fails the Singlish requirement. Do not promote the checkpoint or claim mT5 matches Gemma. This useful pilot signal warrants seed1/2 replication under the frozen recipe, with no new target or prompt tuning and no held-out test access.


## Replication submitted

Job **351130** repeats all three arms for seeds 1 and 2 (six additional runs), using a separate snapshot. All neural source files and target/train/validation hashes match the pilot exactly; only seeds and runner differ. No new target or hyperparameter tuning. Primary outputs remain final step784; each seed requires fresh whole-pair review. Submission is not a completed replication result.


## Replication complete

Job 351130 completed successfully in 12m55s. All six runs reached update784 and saved 50 final pairs. Expected seeds, row counts, 190 training inputs, identical evaluation inputs/references and actual generated-standard anchors were verified across all nine pilot/replication runs. Frozen report hashes: `experiments/shared_intent_v1_replication.json`.

Original versus rewrite received new anonymous whole-pair assistant inspection for seeds1/2 (100 pairs per seed). Aligned-only arms were trained but have not received additional seed1/2 qualitative review, so do not claim replicated isolation of target changes alone.

| Seed | Original meaning / all-three | Rewrite meaning / all-three |
|---|---:|---:|
| 425 | 17 / 8 | 42 / 23 |
| 1 | 21 / 8 | 48 / 28 |
| 2 | 23 / 9 | 48 / 23 |

All counts are out of50 per arm/seed. Means across the three runs:

| Criterion | Original mean per50 | Rewrite mean per50 |
|---|---:|---:|
| context_pass | 33.0 | 33.3 |
| same_meaning_pass | 20.3 | 46.0 |
| singlish_pass | 23.7 | 36.7 |
| all_three_pass | 8.3 | 24.7 |

These are assistant judgments on the SAME 50 reused conversations for each seed, not independent human gold, 150 independent conversations, or final accuracy. The reviewer authored target changes and knows prior controls; hidden model IDs do not ensure independent blinding. Uncertain judgments are not passes, and subjective labels can be inconsistent.

The meaning/style signal repeats, but conversation quality remains roughly unchanged. Recurring context failures include asking a peer who lacks printer knowledge for instructions, reversing who borrows the charger, and replying to an assignment-answer request with an unrelated action. Actual outputs are collected in `artifacts/shared_intent_v1/context_error_review.json` for next work; this artifact is not approved training data. No default model change and no RL run.
