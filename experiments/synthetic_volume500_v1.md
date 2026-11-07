# Synthetic volume experiment

Requested on 2026-11-03: expand beyond the 190 authored training conversations.

## Submitted work

- Pilot: group-account Crimson job 350494; target 24 model-reviewed conversations, budget 48 candidates.
- Full generation: job 350495, dependent on pilot success; target 500 model-reviewed conversations, budget 800 candidates, six-hour limit, one GPU. Fresh output directory `artifacts/volume500_v1/full` on the cluster. Pilot examples are not seeded into this batch.
- Script: `scripts/generate_synthetic_volume.py`; batch wrapper: `scripts/v5_critic.sbatch`.

These are submission details, not evidence that either job finished. Read the remote manifest for current counts and final status. An insufficient-approved result must not proceed to training.

## Quality gates and limits

Generate paired STANDARD/LOCAL replies across six speech-act groups and varied incoming registers. Reject malformed examples, explicit role/meaning problems identified by review, duplicate protected messages, lexical near duplicates, and examples exceeding the actual mT5 token limits. Do not read the held-out test set. Plausible new details remain allowed under the user's selectable-draft policy.

Writer and reviewer are fresh calls to the **same frozen Gemma model**, so their agreement is not independent validation. Model approval is provisional. In the first ten approved pilot examples, assistant inspection found a meaning mismatch in `SYN-volume500-v1-pilot-002-00`: STANDARD promises to report back, whereas LOCAL only says it will check the file. This illustrates a missed error; do not describe these examples as fully audited or human reviewed. Pilot examples are excluded from the full batch seed. Audit the full batch before admitting it to training, and replenish exclusions if needed to reach 500 usable additions.

## Initial comparison plan (superseded by frozen design below)

Use a nested 190-example subset and the full 500-example reviewed pool: 190 original versus 380 versus 690 conversations. At two style rows per conversation plus 180 context rows, these correspond to 560, 940 and 1,560 training rows.

Keep the same base model, LoRA configuration and three seeds. Give augmented data sufficient training exposure and include a longer-trained original-data control to distinguish extra compute from extra data. The earlier 59-example, fixed-280-update result does not resolve this volume question. Select checkpoints using validation only; inspect reply quality as well as validation loss. Do not replace the selected default model without evidence of improvement.


## Assistant audit completed

All 500 conversations were inspected by the Codex assistant, with the user-approved selectable-draft policy. This was neither human review nor a blind independent evaluation. Original generated examples and automatic verdicts are preserved under `artifacts/volume500_v1/full/`.

The explicit audit/revision script is `scripts/curate_volume500_v1.py`; its full before/after ledger is `artifacts/volume500_v1/full/assistant_audit.json`. Final training corpus: `data/synthetic/volume500_v1_reviewed.jsonl`, SHA256 `401131bf5998c8ab040c2ee93827791e7f12089834b59ce7129241d5e29094df`.

- 169 items received an edit: 120 paired-target edits, seven incoming-message edits, one history logic edit, plus removal of redundant final peer history turns. Categories overlap.
- Many changes tightened wording or aligned commitments/certainty; 120 edits must not be interpreted as 120 equally severe errors or an independently measured failure rate.
- The initial raw scan found 60 repeated incoming/final-history messages. Only duplicate final **peer** turns were removed; user turns are not automatically removed.
- A fresh full generation restarted the same initial seeds as the diagnostic pilot, so early examples were reproduced even without loading pilot data. Pilot examples are not an independent corpus, test, or additional training count.
- Full-corpus loader validation passed: 500 items, 1,000 style rows, no exact normalized protected training/validation message overlap. This is not proof of semantic independence or response quality.

## Frozen training design

Recipe: `experiments/synthetic_volume500_v1_recipe.json`. Four arms, seeds 425/1/2, same frozen base and LoRA: original data at 280 updates; original data at 784 updates; +190 nested examples at 472 updates; +500 at 784 updates. The augmented arms receive eight full epochs. Patience 100 prevents early stopping from reducing exposure; checkpoint selection still uses the minimum validation loss. The longer original-data arm has more checkpoint selection opportunities, and equal optimizer steps do not imply identical token counts/FLOPs.

The isolated cluster snapshot checks source/data hashes, nested subset membership, protected overlaps, row counts, and the actual tokenizer's 128/48 input/target limits before training. No held-out test reads and no default-model replacement. Local trainer/loader checks: 37 tests passed.

Training submitted as Crimson group-account job **350866**, observed RUNNING on `aloha`, one NVIDIA RTX 3090 (24 GB VRAM). Twelve runs total: four arms × three seeds. Submission is not a completed result.


## Completed results

Job 350866 completed successfully in 20m35s. All twelve reports retrieved and expected seeds, update counts, training row counts, and 50 validation outputs per run verified. Detailed report hashes and means: `experiments/synthetic_volume500_v1_results.json`.

| Arm | Mean best validation loss |
|---|---:|
| control8 | 2.5068 |
| control_long | 2.4959 |
| plus190 | 2.5022 |
| plus500 | 2.5170 |

Single-seed (425) anonymized assistant inspection of long-control versus +500: 16 prefer long-control, 11 prefer +500, 40 ties, 27 both bad, six identical. This is exploratory assistant judgment, not independent human review, and not an overall error rate; identical replies were not separately assessed. These results do not establish a +500 quality benefit. Validation remains reused development data. Default model unchanged.

Next recommendation: inspect context/role and non-answer errors, compare against an instruction-tuned baseline under the same review protocol, and establish reward-scorer reliability before online RL. Do not infer a MoE/LoRA or RL result: neither experiment has been run.
