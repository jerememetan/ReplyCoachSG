# Final report inclusion notes

User requirement, recorded 27 October 2026: **include the training and refinement
of Role E's V6 reranker in the final project report.** This is a report-writing
reminder and evidence index, not the completed final report.

## Planned section: Reranker training and refinement

Final demo direction, recorded 29 October 2026: use **V6 (Qwen + DailyDialog)**
for the shared reranker: Qwen3-Reranker-0.6B, DailyDialog adaptation, then
reviewed-data LoRA refinement (selected refinement epoch 2). Identify earlier
MiniLM-based V6 runs as baselines so the report does not confuse their architecture
with the final demo choice. This user-selected direction does not establish a
quality advantage over V5; retain the matched comparison and its limitations.
Application integration now defaults All models to **V6 (Qwen + DailyDialog)**,
with older MiniLM checkpoints selectable for comparison. It preserves source
proposals on checkpoint failure and exports the actual adapter identity and
yes-minus-no score kind. The overall app still opens in Retrieval comparison.
Integration evidence: 75 targeted tests passed and a real app request verified
all four source models, the final adapter hash, both selected replies and the
preview click behavior. The 7pm confirmation example returned “Yes, 7pm is fine.”
and “Yep, 7pm is fine.” This verifies operation, not general quality improvement.

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

- [Frozen 60-item final evaluation](../final_test_evaluation.md): authorized on
  2 November, all 60 comparisons exported, and judging resumed on 3 November from
  310/4,800 successful ratings. Keep this separate from validation and wait for
  the completed result receipt before adding held-out quality tables.

- [DailyDialog → reviewed-data Qwen adaptation](../dailydialog_reranker_training.md):
  user-run GPU training completed and results imported/verified on 29 October.
  Adaptation ran two epochs, selecting epoch 1 (loss 0.249795 on its own external
  validation data). Refinement ran four epochs under the 12-epoch cap, selecting
  epoch 2 (loss 0.341845 versus 0.349055 reviewed-only on the same project pairs).
  The selected stage-1 adapter was verified as stage 2's initialization; base,
  adapter, pair and validation hashes passed. The [matched replay](dailydialog_reranker_comparison_20261029.md)
  completed on the same 400 proposals and 4,000 existing AI-persona ratings:
  new Qwen 3.634 Standard / 3.988 Local versus previous Qwen 3.640 / 3.946 and
  V5-only 3.666 / 4.110. New-minus-previous appropriateness differences are
  −0.006 [−0.088, +0.088] Standard and +0.042 [−0.152, +0.232] Local. Only ten
  reply texts changed: five rated higher, four lower and one tied. No useful
  overall gain is established; the replay itself did not change demo defaults.
  The user subsequently selected this checkpoint for the V6 demo integration above.
  Duplicate-text
  sensitivity agrees. All ten choices in the five repeated-speaker conversations
  are unchanged. Include the charger borrower/lender reversal incorrectly rated
  5/5 by the existing AI judge as a concrete evaluation limitation.
  Explain the source-turn assumption, weak labels, exact-overlap
  filtering, separate demo selection decision and untouched project test set. A lower pair loss
  does not establish improved reply quality or isolate the cause of any gain.
  The first user-run attempt reached at least 300 adaptation updates, then failed
  on nonfinite gradients before completing epoch 1. The 29 October AMP recovery
  correction is documented in that experiment note. The completed corrected rerun
  recovered once after update 311, lowering the scale 65,536→32,768; refinement
  recorded no overflow. Keep the failed run distinct from the verified completed
  rerun, and numerical stability distinct from reply-quality evidence.

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
- [Maximum-12-epoch Qwen follow-up](../qwen_reranker_12_epochs.md): the user
  requested a longer run because validation loss was still falling at epoch 4.
  This separate experiment kept data and other settings fixed and used patience 2.
  It stopped after six epochs (438 updates), retaining epoch 4 (292 updates),
  as validation loss rose at epochs 5 and 6. Its completed replay made the same
  100 choices as the earlier four-epoch adapter: 3.640 Standard / 3.946 Local.
  Include the training curve and absence of measured selection improvement.
  Report the requested cap, actual epochs and selected checkpoint separately;
  epochs 7–12 were not executed, and pair loss is not reply-quality evidence.
- The linked reports identify private Kaggle resources and ignored local raw
  evidence. Do not publish credentials, checkpoints or private raw logs in the
  report/repository merely to document the experiment.
