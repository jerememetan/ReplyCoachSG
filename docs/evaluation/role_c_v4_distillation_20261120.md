# V3/V4 validation and SEA-LION distillation results

This note supports **Table 3.2. V3/V4 development validation diagnostics and distillation results** in the final report. It combines the original Role C comparison with the two subsequent V4 distillation experiments. The companion [JSON summary](role_c_v4_distillation_20261120.json) preserves the aggregate values and provenance.

## Development validation results

All rows use the same 50 team-authored validation conversations, with one STANDARD and one LOCAL output per conversation. They are development results, not a new held-out test evaluation.

| Model | Validation CE | STANDARD token F1 | LOCAL token F1 | Different style pairs /50 |
|---|---:|---:|---:|---:|
| V3 baseline | 4.2023 | 0.1336 | 0.0879 | 28 |
| Original V4 | 4.1619 | 0.1168 | 0.0918 | 34 |
| V4 first distillation | 4.1503 | 0.1072 | 0.0922 | 35 |
| V4 A: 197 diverse synthetic targets | 4.1614 | 0.1274 | 0.1057 | 32 |
| V4 B: 292 diverse synthetic targets | 4.1632 | 0.1128 | 0.0970 | 30 |

CE is token-averaged, teacher-forced cross-entropy. Token F1 measures overlap with the human reference, rather than appropriateness or understanding. Different output strings do not establish correct style or equivalent meaning. Original task metrics were recorded on CPU; the distillation comparison recomputed V4 on CUDA. Original V4 CE was 4.161940 on CPU and 4.161924 on CUDA, which both round to the table value.

Vocabulary sizes are 5,251 for V3/original V4, 5,255 for the first distillation, and 5,393 for A/B. Vocabulary changes affect softmax normalisation and reference unknown tokens, so CE across those groups is not directly comparable. A/B share a vocabulary expanded from all accepted synthetic training text, including text whose targets are not selected for A.

## Training and synthetic data

V3 is an LSTM encoder-decoder; V4 adds Luong dot-product attention. Both use 64-dimensional embeddings and 128-unit LSTMs. Five-epoch DailyDialog checkpoints were task-adapted using 190 human training conversations (380 style-specific targets), Adam at 0.001, batch 32, seed 425, a 60-epoch maximum and patience 10. V3 selected epoch 10 and V4 epoch 8. A historical audit found 99 complete-dialogue overlaps between the public DailyDialog training and validation files; historical public validation loss is therefore not independent. The task validation set is separately team-authored.

The **14 November first distillation** used SEA-LION to generate replies for the existing 190 training conversations. Screening retained 108 pairs; removing duplicate source/target examples left 197 additional targets. It added replies rather than new task contexts.

The **20 November diverse-scenario experiment** generated replies for 438 candidate inputs: 357 DailyDialog training windows and 81 individually AI-authored scenarios. Deduplication and scope checks preceded generation. Reply screening retained 146 pairs (33.3%): 107 DailyDialog and 39 original scenarios, covering all nine scenario categories. These are explicitly synthetic training examples, not human gold data. Public-corpus speaker roles are inferred from two-party turn alternation and recorded as inferred. Screening used automatic checks and AI-assisted inspection, not independent human review. Forty random accepted pairs were exported for pending human review.

A selected exactly 197 targets from 122 new scenarios (99 STANDARD, 98 LOCAL), matching the first round's target count. B used all 292 targets from the 146 accepted pairs. One target is one reply in one style, not a complete dialogue pair.

Both arms started from the **original V4 task checkpoint**, retaining the LSTM/Luong architecture. Training and inference shared explicit `<peer>`, `<user>`, `<other>` and `<incoming>` markers plus the existing style tokens. Adam used 0.0003, seed 425, batches of 16 human and 16 synthetic targets sampled with replacement, 12 updates per epoch, a 30-epoch maximum and patience 6. Source/target limits remained 64/40 tokens. Both ran eight epochs and selected epoch 2 by human validation CE. Used markers received nonzero gradients and changed in the selected weights. The first-round markers-only control selected step 0, so it did not demonstrate a benefit from learned marker adaptation alone.

## Interpretation and uncertainty

A's mean F1 across styles increased by 16.9% relative to the first distillation and 11.8% relative to original V4. Paired bootstrap resampling uses 50 conversations, 2,000 samples and seed 425, retaining both styles within each sampled conversation.

| Comparison | Mean F1 difference | 95% paired interval | Exploratory 90% interval |
|---|---:|---|---|
| A minus first distillation | +0.01681 | [-0.00052, +0.03622] | [+0.00169, +0.03310] |
| A minus original V4 | +0.01226 | [-0.01635, +0.03731] | [-0.01050, +0.03381] |

The 90% intervals are a post-result exploratory supplement; the original 95% intervals remain reported. The results show an encouraging reference-overlap signal for new scenarios at a matched target count, without establishing a reliable overall quality advantage over original V4. This is a single-seed study on reused development validation. B did not improve further; the fixed update budget also gives each B target fewer expected exposures, so the comparison does not isolate data quantity.

Context errors persist. For `where you want eat?`, A/B answer `i am going to the library`, overlapping the reference's food-court-near-library wording without properly answering the food question. Training loss falls while validation loss rises after epoch 2. Teacher target quality, generalisation and the token-prediction objective are possible contributors; limited student capacity remains a hypothesis without a size ablation.

The original V4 remains the default. These experimental checkpoints were not substituted into the earlier six-system judge evaluation. For that separate evaluation, see [the pipeline findings](current_pipeline_20261108.md).

## Evidence and verification

Saved checkpoints were independently reloaded to reproduce all 100 outputs per V4 system (400 outputs in the second-round verification). No distillation job read `data/test/test.jsonl` or used validation examples as synthetic generation seeds or training targets. Validation inputs were used only for development evaluation and exclusion checks against duplicate candidate inputs.

Raw evidence stays in ignored local experiment directories:

- `artifacts/role_c_runs/20261020_colab_handoff/`: original V3/V4 comparison and task training reports.
- `artifacts/role_c_runs/20261114_distillation/cluster_results/`: first-round report, protocol, comparison, verification and checkpoints.
- `artifacts/role_c_runs/20261120_distillation_diverse/cluster_results/`: source/dedup records, teacher replies, screening, training, comparison, validation outputs and verification.
- `artifacts/role_c_runs/20261121_report_section/bootstrap_supplement.json`: exploratory 90% intervals recomputed from saved outputs.

Second-round Slurm jobs: **372646** (26-input pilot), **372663** (bulk teacher generation), **372706** (student training, comparison and independent reload verification), using one RTX 3090 per GPU job. Large checkpoints, raw dialogue outputs and logs are distributed separately; the repository summary alone does not contain everything needed to rerun inference.
