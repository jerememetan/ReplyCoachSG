# Qwen LoRA: maximum 12-epoch follow-up

28 October 2026. The user approved extending the earlier four-epoch experiment
to a maximum of 12 epochs, with early stopping after two consecutive epochs
without lower validation pair loss. The earlier validation loss was still
falling at epoch 4; longer training is worth measuring, not assumed better.

## Controlled protocol

Start from the same pinned pretrained Qwen weights and seed 425. The earlier
run saved an adapter but not optimizer/RNG state, so this is a new controlled
training run, not exact optimizer-state resume. Preserve the four-epoch adapter.

Keep the same instruction, 512-token budget, AdamW 2e-5, weight decay 0.01,
microbatch 2 / accumulation 8, LoRA rank 8 / alpha 16 / dropout 0.05 and q/v
projections. Only the maximum epoch budget and early-stopping rule change.

Training data and labels are byte-identical: 1,160 pairs from the same 190
training conversations, including the 20 explicit reviews. Validation uses the
same 300 pairs / 50 conversations. Retain the 60 held-out test items untouched
and all 4,000 existing AI ratings as evaluation-only records. Do not introduce
new labels, augmentation or a new data split.

Save the checkpoint with the lowest validation pair loss, not automatically the
last epoch. Record completed epochs, requested cap, actual updates, patience,
stop reason and weight/data hashes. Stop after two non-improving validation
checks; report honestly if fewer than 12 epochs complete. The independent T4
notebook gets a 5,400-second timeout so a full run fits the observed earlier
three-minute epochs plus setup. This is a service timeout, not an ETA.

Compare only the selected checkpoint against pretrained Qwen, the preserved
four-epoch adapter and V5-only on the identical 400 saved proposals and existing
AI ratings. Include paired conversation-bootstrap intervals and changed-choice
examples. No tuning or epoch selection using those AI ratings. Keep the demo
unchanged while evaluating, and document mixed or negative outcomes as well.

## Baselines

| Selector | Standard appropriateness /5 | Local appropriateness /5 |
| --- | ---: | ---: |
| Pretrained Qwen | 3.578 | 4.044 |
| Qwen LoRA, four epochs | 3.640 | 3.946 |
| V5-only | 3.666 | 4.110 |

See the [original Qwen report](qwen_reranker_experiment.md) for initialization,
software checks, private resources and limitations. These are reused development
validation results from correlated AI-persona ratings, not held-out human tests.

## Private resources and status

- [Private dataset](https://www.kaggle.com/datasets/tanjingminjereme/replycoach-qwen-pairs-20261028-12).
- [Private GPU notebook](https://www.kaggle.com/code/tanjingminjereme/replycoach-qwen-lora-20261028-12).
- Local package/evidence: `artifacts/reranker_qwen/kaggle_20261028_12/`.

The full suite passed 419 tests in 91.62 seconds, including epoch-budget,
patience reset, best-checkpoint and notebook-argument checks. The training
bundle is verified identical to the preceding experiment. The private dataset
and T4 notebook (both version 1) have been created and read back through the
official authenticated Kaggle MCP, confirming privacy and GPU assignment. The
Docker image matches the four-epoch GPU run. Training is running; no completed
result or adapted reply-quality claim is made yet.
