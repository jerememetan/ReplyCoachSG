# Final report inclusion notes

User requirement, recorded 27 October 2026: **include the training and refinement
of Role E's V6 reranker in the final project report.** This is a report-writing
reminder and evidence index, not the completed final report.

## Planned section: Reranker training and refinement

Place the method in the model/methodology section and the measured comparison
in experiments/results. Cover both the initial fine-tuning and the subsequent
augmentation experiment; retain the mixed result rather than reporting only
the improvement over frozen scoring.

- Explain the shared MiniLM cross-encoder: it scores conversation, requested
  reply style and candidate reply, then selects among BM25/MiniLM/V4/V5 proposals.
- Describe supervision from 190 reviewed training conversations, both reply
  styles, and user-approved mismatched replies as weak negatives. The original
  bundle has 1,140 training pairs and 300 validation pairs from 50 conversations.
- Record the pinned pretrained initialization, AdamW settings, GPU training,
  four completed epochs, best-checkpoint selection by validation loss and
  provenance checks. The original run made 288 updates and retained epoch2.
- Describe refinement using a conservative snapshot of the existing ABBR
  mapping: seeded whole-word replacements in conversation inputs only, up to
  three additional distinct variants, probability0.35, seed 425. Preserve replies,
  roles, style instructions and the original validation split. The resulting
  2,556 pairs represent the same 190 underlying conversations, not 2,556
  independently reviewed examples. The augmented run made 640 updates and
  retained epoch2; optimizer-step budgets differ at the same epoch count.
- Report the matched replay on 400 saved proposals / 50 validation conversations
  using the same 4,000 existing AI-persona ratings, with no new judge calls.

| Selector | Standard appropriateness /5 | Local appropriateness /5 |
| --- | ---: | ---: |
| Frozen, original input format | 3.060 | 3.796 |
| Frozen, style-aware input control | 2.970 | 3.810 |
| Fine-tuned V6 without augmentation | 3.528 | 3.878 |
| Fine-tuned V6 with augmentation | 3.442 | 3.926 |
| V6 with 20 additionally reviewed alternatives | 3.508 | 3.930 |
| Fixed V5 | 3.666 | 4.110 |

Discuss the Standard improvement over frozen scoring, unresolved Local gain,
and mixed augmentation changes. Neither augmentation nor hybrid selection has
established a gain over the best generator. Keep the paired uncertainty
intervals and the initial decision to retain the original checkpoint. On
28 October the user explicitly requested the 20-review checkpoint as the
All models demo default; distinguish that demonstration choice from evidence
of better quality. Original V6 and Frozen MiniLM remain selectable.

Limitations must include potentially valid weak negatives, small independent
sample size, validation reuse for checkpoint selection and development analysis,
one augmentation seed, unequal update budgets and correlated AI-persona ratings.
These are not independent human-survey results or held-out test results. The
60 test items were not used in these experiments. Refresh the status and figures
against any later experiments before writing the final report.

## Evidence to use when writing

- [Original V6 training, settings, losses and comparisons](../../roles/E_rerank_demo/V6_TRAINING.md).
- [Augmentation design, completed Kaggle run, controls and measured results](../reranker_augmentation.md).
- [Earlier four-system comparison](four_system_comparison_20261026.md).
- [Reranker choice diagnosis](reranker_diagnosis_20261027.md): all100 decisions
  reproduced, concrete role/time failures, candidate-pool limits, judge concerns
  and the training-gap hypothesis. Include representative errors in discussion.
- [Reviewed difficult-comparison pilot](../reranker_hard_examples.md): distinguish
  the 20 completed training-label reviews (13 suitable, 7 unsuitable) from blind
  human evaluation. GPU training completed 292 updates; best epoch 3. The matched
  validation replay gives mixed, unresolved changes and no established overall
  gain. Include this completed refinement experiment and its limitations.
- [Qwen pretrained and LoRA comparison](../qwen_reranker_experiment.md): include
  the subsequent architecture/instruction experiment. Raw Qwen has completed
  the matched replay (Standard 3.578, Local 4.044); gains versus reviewed MiniLM
  remain unresolved. GPU LoRA training completed four epochs / 292 updates,
  retaining epoch 4. Its completed matched replay scores 3.640 Standard and
  3.946 Local: Standard +0.062 versus raw Qwen, Local -0.098, with intervals
  including zero. Include the mixed result, weak-label limitations, real greeting
  diagnostic, observed runtime, private provenance and preserved checkpoints.
- The linked reports identify private Kaggle resources and ignored local raw
  evidence. Do not publish credentials, checkpoints or private raw logs in the
  report/repository merely to document the experiment.
