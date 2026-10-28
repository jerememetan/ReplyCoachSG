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
and T4 notebook have been created and read back through the official
authenticated Kaggle MCP, confirming privacy, GPU assignment and the attached
dataset. Dataset version 1 is unchanged. Notebook version 1, submitted through
MCP, lacked the requested data attachment and stopped before training at the
exact-one-bundle assertion. Version 2 was resubmitted through the proven CLI
path; its data attachment was explicitly verified. The Docker image matches the
four-epoch GPU run. Training, checkpoint verification and the selected-checkpoint
reply replay are complete; see the measured comparison below.

## Completed training and early stopping

The requested cap was 12, but the agreed patience-2 rule stopped the run after
**six completed epochs**. Validation loss worsened at epochs 5 and 6, so the
saved best adapter is **epoch 4**. There were 438 optimizer updates across the
run; the selected epoch-4 checkpoint received 292 updates.

| Epoch | Training pair loss | Validation pair loss | Total updates |
| --- | ---: | ---: | ---: |
| 1 | 0.42883 | 0.38862 | 73 |
| 2 | 0.36634 | 0.36886 | 146 |
| 3 | 0.33678 | 0.36352 | 219 |
| 4 | 0.30219 | **0.34916** | 292 |
| 5 | 0.26558 | 0.35620 | 365 |
| 6 | 0.22682 | 0.37653 | 438 |

Falling training loss alongside rising validation loss is consistent with
overfitting on this labelled-pair task. This run did not execute epochs 7–12,
so it cannot establish what an uninterrupted full-12-epoch trajectory would do.
It does show that the pre-agreed stopping/selection rule still chooses epoch 4.
Pair loss alone does not establish conversational reply quality.

The downloaded adapter/base file hashes and the actual 1,160 GPU training rows
were checked. Data, initial adapter hash, base model/revision, prompt, seed,
learning rate, batch/accumulation, LoRA settings and package versions match the
earlier run. The first four losses are close but not bitwise identical, and the
new epoch-4 adapter tensors differ; therefore a fresh reply replay is required.
Treat this as a fresh GPU replication with observed numerical variation, not
exact optimizer-state continuation or evidence from an independent data split.

The base parameters remained unchanged. Training/validation/checkpoint saving
took 1,001.87 seconds after setup/model loading. The selected adapter is at
`artifacts/reranker_qwen/kaggle_output_20261028_12/qwen_lora/best/`, with file hash
`ad95d31f3d4415feaa93d8c534c98d27d6f68b82338c5a8875a159bfc88b1dda`.

## Completed matched reply-selection comparison

The actual loaded adapter tensor hash was verified before replaying the same
400 saved proposals / 50 validation conversations. Source-export, packet,
prompt and existing 4,000 AI-rating hashes match the previous runs. No new judge
calls, human labels or test-set examples were used. Both reply styles were
selected separately with the same CPU precision and token budget.

| Selector | Standard appropriateness /5 | Local appropriateness /5 | Standard would-send | Local would-send |
| --- | ---: | ---: | ---: | ---: |
| Raw Qwen | 3.578 | 4.044 | 58.8% | 72.4% |
| Original four-epoch Qwen adapter | 3.640 | 3.946 | 60.8% | 70.0% |
| 12-epoch cap, best epoch 4 | 3.640 | 3.946 | 60.8% | 70.0% |
| V5-only | 3.666 | 4.110 | 61.2% | 76.8% |

**All 100 selected candidate IDs match the earlier four-epoch adapter**: zero
Standard changes and zero Local changes. Paired differences in all four reported
metrics are zero, with conditional 95% conversation-bootstrap intervals [0, 0]
(20,000 resamples, seed 425). This follows from identical selections on the frozen
packet; it does not mean the weights are identical or that the models behave
identically on every possible conversation. Actual tensors and logits differ
slightly, consistent with the observed numerical variation between GPU runs.

The disclosed greeting pair retains the appropriate ordering: “I'm fine, thanks”
scores 2.656 versus -2.847 for the unrelated arrival reply. This diagnostic was
not inserted into training. Observed CPU replay time was 536.24 seconds, with
39.14 seconds of loading/integrity/diagnostic work. As before, these timings are
not a controlled latency benchmark.

**Conclusion:** extending the budget with the agreed patience-2 rule did not
produce a better validation-selected checkpoint or change the saved reply choices.
Epoch 4 remains selected under this protocol. This is not a test of uninterrupted
training through epoch 12, and later recovery beyond epoch 6 remains unobserved.
Repeated development validation, weak negatives, one seed and correlated
AI-persona ratings limit claims about general conversational quality. Neither
these results nor the prior experiments establish superiority over V5-only.
The final held-out 60-item evaluation remains separate work.

Earlier checkpoints and the demo remain unchanged. No model was silently promoted.

## Evidence and reproduction

Ignored evidence under `artifacts/reranker_qwen/` includes:

- `kaggle_20261028_12/`: allowlisted package, source hashes, privacy and attachment
  readbacks, completion receipt and the version-1 attachment-failure log.
- `kaggle_output_20261028_12/`: GPU epoch log, history, exact training rows and
  selected adapter/manifest.
- `replay_lora12_20261028_01/result.json`, `comparison12_20261028.json` and
  `lora12_greeting_20261028.json`: matched replay, paired comparison and loaded
  tensor-hash check.
- `qwen_12cap_training_curve_20261028.png`: verified training/validation plot.
- `full_tests_20261028_12.txt`: 419-test full-suite output.

```powershell
.\.venv\Scripts\python.exe scripts/package_qwen_reranker.py --bundle artifacts/reranker_v6/kaggle_20261027_reviewed01/dataset --output artifacts/reranker_qwen/new_12_epoch_package --owner YOUR_KAGGLE_SLUG --tag YYYYMMDD-NN --epochs 12 --early-stopping-patience 2
.\.venv\Scripts\python.exe scripts/replay_shared_ranker.py --packet experiments/judging/four_systems_20261026_01/main_packet --qwen-model artifacts/reranker_qwen/pretrained --qwen-adapter artifacts/reranker_qwen/kaggle_output_20261028_12/qwen_lora/best --out-dir artifacts/reranker_qwen/new_12_epoch_replay
```

Use new output directories and retain the private dataset/checkpoints separately.
The 12-epoch CLI settings are explicit; default Qwen training remains four epochs.
