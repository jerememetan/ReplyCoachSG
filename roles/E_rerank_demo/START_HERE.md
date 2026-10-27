# Role E — Reranker, integration, demo

## Current priority updated 27 October 2026

The proposal's V6 fine-tuning experiment is complete. See
[V6 training and validation](V6_TRAINING.md) for the private Kaggle run, installed
checkpoint, actual losses, frozen-versus-trained comparisons and limits. The
demo has an explicit Fine-tuned V6 shared-selector option. It improves Standard
selection over the frozen reranker on development data, but does not establish
an advantage over V5-only; keep it experimental. The 60-item held-out evaluation
and independent teammate sanity checks remain separate work.

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
