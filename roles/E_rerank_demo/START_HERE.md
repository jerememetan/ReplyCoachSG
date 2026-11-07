# Role E — Reranker, integration, demo

## Role D checkpoint handoff — 8 November 2026

The exported generator references are documented in the [mT5 A model card](../D_transformer/mt5_reference_A_MODEL_CARD.md) and [SEA-LION repair25 model card](../D_transformer/sealion_repair25_MODEL_CARD.md). The [checkpoint handoff guide](../../artifacts/model_handoff_v1/README.md) records the archives, loader and checksums; full model weights are shared separately from Git. These are experimental generator references and do not change the shared selector or demo defaults below. Read [the current transformer handoff](../../docs/transformer_resume.md) for the latest generator results. Earlier dated entries describe their own experiment scope.

## Current priority updated 29 October 2026

User direction on 29 October: the planned final demo **V6 reranker** is Qwen3-Reranker-0.6B
with DailyDialog adaptation followed by reviewed-data LoRA refinement. The selected
adapter is `artifacts/reranker_qwen/dd_result_20261029/stage2_reviewed/best`.
Earlier MiniLM-based V6 checkpoints remain comparison baselines. This is a demo
selection decision, not evidence of superiority over V5. All models now defaults
to **V6 (Qwen + DailyDialog)**. The older MiniLM checkpoints remain selectable
under explicit MiniLM labels. See the README for private checkpoint locations
and environment overrides. Restart Streamlit after updating the source.

The user-run [DailyDialog → reviewed-data Qwen experiment](../../docs/dailydialog_reranker_training.md)
completed GPU training and local artifact verification on 29 October. DailyDialog
ran two epochs and retained epoch 1; reviewed refinement ran four epochs and
retained epoch 2 (validation pair loss 0.341845 versus 0.349055 reviewed-only).
The overflow correction recovered once at update 311. Adapter/base hashes,
finite tensors, stage continuation and unchanged project data were verified.
The [matched reply-selection comparison](../../docs/reports/dailydialog_reranker_comparison_20261029.md)
is complete: new Qwen scores 3.634 Standard / 3.988 Local against previous Qwen's
3.640 / 3.946 and V5-only's 3.666 / 4.110. Appropriateness intervals include zero;
no useful overall gain is established and no demo promotion is made. The five
repeated-speaker conversations retain all ten previous choices. Report the
concrete improvements, regressions and AI-judge role error. Results are separate from existing checkpoints and
keep the 60 project test items untouched. Include the failed attempt, correction,
completed training and mixed comparison in the final report.

The proposal's V6 fine-tuning experiment is complete. See
[V6 training and validation](V6_TRAINING.md) for the private Kaggle run, installed
checkpoint, actual losses, frozen-versus-trained comparisons and limits. The
demo has an explicit Fine-tuned V6 shared-selector option. It improves Standard
selection over the frozen reranker on development data, but does not establish
an advantage over V5-only; keep it experimental. The 60-item held-out evaluation
and independent teammate sanity checks remain separate work.

Final report requirement: include reranker training and refinement, including
the augmentation experiment and its mixed result. The planned section and
evidence index are in [final report notes](../../docs/reports/final_report_notes.md).

The [reviewed difficult-comparison pilot](../../docs/reranker_hard_examples.md)
is prepared in the existing team reviewer under **Reranker comparisons**. Its
20 training-only alternatives now have human decisions (13 suitable, 7 unsuitable).
A separate Kaggle experiment and matched validation replay are complete. The
additional labels give mixed, unresolved changes (Standard 3.528→3.508,
Local 3.878→3.930). On 28 October the user requested this reviewed checkpoint
as the All models demo default, labelled **Fine-tuned V6 (20 reviews)**. Original
V6 and Frozen MiniLM remain selectable. This is a demonstration choice, not an
established quality gain. Follow the linked report for losses, intervals and
limits. Miner scores and AI evaluation ratings are not training labels.

The user subsequently approved a [Qwen pretrained/LoRA experiment](../../docs/qwen_reranker_experiment.md).
Raw Qwen has completed the saved-proposal replay and fixes the disclosed greeting
ordering; private LoRA training and matched testing are complete. LoRA scores
3.640 Standard / 3.946 Local versus raw Qwen's 3.578 / 4.044, a mixed and
unresolved change. Both handle the greeting ordering. See the report for intervals
and remaining role errors. Reviewed MiniLM was the demo default before the
29 October Qwen + DailyDialog integration above.

The subsequent [12-epoch-cap follow-up](../../docs/qwen_reranker_12_epochs.md)
is also complete. Patience 2 stopped after six epochs and retained epoch 4;
all 100 saved reply choices match the earlier four-epoch adapter. More training
under this protocol produced no better validation-selected checkpoint. Preserve
this result and its limitations in the final report.
## Historical Role D evaluation handoff — 3 November 2026

This entry records the generator handoff on that date; it does not supersede Role E's later frozen-evaluation work or results.

D’s bounded training ablations are complete; retain the existing supervised V5
checkpoint. See [the consolidated results](../../experiments/v5_ablation_summary_20261103.md).
No new experimental adapter was promoted, and the final test set remains locked.

The user permits plausible new personal/factual details in selectable drafts.
The evaluator-only and shared rubrics now reject explicit contradictions/role
reversals, rather than missing evidence alone. Use fresh matched calibration/main
folders after syncing the updated rubric; old judge runs retain their original
rubric and must not be merged. The runner records rubric text in resume checks.
Fresh P01 policy diagnostic matched 6/6 reviewed examples; this is not main-panel
accuracy. See [the check](../../experiments/eval_policy_20261103.md).

The exporter input check passed for 190 bank items/50 validation queries without
loading models. Real generator/hybrid exports, main ten-persona ratings and demo
proof remain pending. Source/rubric changes are local workspace changes; if running
on the cluster, sync the updated evaluator rubric or use its frozen calibrated
copy in `~/ReplyCoachSG/experiments/eval_policy_20261103/rubric.md`. Do not accidentally
resume the historical cluster checkout’s old-rubric ratings.


## Current priority updated 18 October 2026

The [judge experiment report](../../docs/judge_pilot_experiments.md) records the
Qwen/Hermes/Gemini diagnostics, Gemma selection, completed 200-rating pilot,
known errors and remaining evaluation work.

Start the complete local evaluation pipeline with
`.\.venv\Scripts\python.exe scripts/run_judging.py` after Python and judge setup.
The standard judge is local Gemma4 12B through Ollama. Install with
`ollama pull gemma4:12b`; ignored `.env` uses `OLLAMA_MODEL=gemma4:12b` and a
loopback `OLLAMA_BASE_URL`. No API key or hosted judge is used. Thinking is off,
with 4,096 context tokens and up to 2,048 output tokens.
Use `--calibrate` first for one existing persona (20 ratings by default), with
readable per-reply context and explanations, before a full panel. A successful
run does not establish judge quality. Keep the same judge, settings, rubric,
personas and conversations across systems, manually spot-check outputs and report
known context/style biases. These are AI comparisons, not human feedback.
It exports BM25 M0/M2, prepares five shared conversations and judges them using
the CSV personas. It prints a fresh output folder and resume command; add
`--check-only` for a no-write/no-model preview. The individual tools below remain
available for advanced/manual use.

The export-to-judge handoff is available as `scripts/prepare_judging_inputs.py`:
validate matched development exports and write anonymous `judge_inputs.jsonl`
plus a separate `PRIVATE_mapping.jsonl`, without manual JSON edits or model calls.
Use `--check-only` first and a fresh folder under ignored `experiments/judging/`.
Jereme/E now also owns the judging workflow. Keep both files local; judges receive
only public task text. Run `scripts/judge_replies.py PACKET_FOLDER`, using the
ignored `.env` for the local judge model/address and `data/evaluation/personas.csv` for ten
initial profiles. Each scores appropriateness, naturalness and style fit (1–5)
and a boolean would-send vote. The terminal/CSV summarise by model settings and
style. Use `--check-only`, `--resume` and `--summary-only` as needed. Pilot-check
the local judge; full evaluation results are not complete. Preserve identical
item/style coverage, anonymise system
identities before judging and keep the private mapping separate. Do not send
retrieval/reranking scores to the judge or treat them as quality ratings.

When C/D supplies a real model, verify genuine retrieval-only, generator-only and
hybrid outputs. With B, prepare 20 team-written diagnostic/demo conversations,
including longer context. With A, later provide a separate guarded final-test
output route; do not bypass the development exporter's test guard. See
`docs/role_handoffs.md`, `docs/split_and_rating_protocol.md` and `docs/week_plan.md`.

The original Week 4 outline below is historical; it is not current completion.

## This week (Week 4)

Shared candidate format lives in `src/replycoach/schema.py` (`Candidate`).

Demo:

```powershell
python -m replycoach.demo_cli
streamlit run src/replycoach/demo_app.py
```

Logging: `src/replycoach/logging_utils.py` → `experiments/logs/runs.jsonl`.

A first V6 baseline already rescores with token overlap (`src/replycoach/rerank.py`). Improve it in Week 6; do not replace the format.

## Next

- Week 5: nicer candidate table in the demo.
- Week 6: use retrieval score + message similarity.
- Week 7: include generator candidates.
- Week 8: full flow with fallback and two replies.
- Week 11: presentation demo on 20 unseen conversations.

## Friday artefact

A screenshot or CLI printout showing STANDARD and LOCAL from V0.
