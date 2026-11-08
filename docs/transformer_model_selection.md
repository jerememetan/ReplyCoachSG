# Choosing the transformer for the demo

Completed findings: [full report](evaluation/transformer_selection_20261108.md)
and [aggregate evidence](evaluation/transformer_selection_20261108.json).
After reviewing the completed run, the user selected SEA-LION 3B for the
interactive All models V5 slot. Frozen evaluation entry points retain original mT5.

Compare the original V5 mT5, the mT5 A handoff and SEA-LION repair25 on the same
50 reviewed validation conversations. Keep the 60 held-out test conversations
outside this selection exercise. The 190-example training/reply bank is checked
for validation overlap but is not retrieved to construct generator replies.

```powershell
.\.venv\Scripts\python.exe scripts/compare_transformer_candidates.py --out experiments/judging/transformer_selection_20261108_01
```

Append `--resume` for the same existing run. Inputs, generator source, model
weights/manifests, rubric, ten personas and local Gemma model identity are frozen
before generation. Stop the Streamlit demo first to free its GPU models. Outputs,
credentials and ratings remain outside Git.

Each fixed provider generates Standard and Local replies using the demo's M2
memory and inferred speaker convention. This comparison uses outputs before
shared reranking or V7 refinement. A failed provider is recorded visibly; its
existing app safety fallback is judged instead of substituting another generator
or dropping the difficult conversation. Generation failures are reported separately.

The existing Gemma4 12B judge, schema, instructions, rubric, options and ten
fictional personas are reused without additional grading criteria. Model names,
scores, failure metadata and reference replies are hidden from the judge. All
300 candidates are shuffled before anonymous IDs are assigned, for 3,000 nominal
candidate/persona ratings.

Exact duplicate judging tasks can reuse ratings. Earlier validation ratings are
eligible only when the full conversation, incoming message, style, reply, persona,
model digest, options, rubric, schema and the judging function match. The earlier
receipt and packet hashes are checked. Every reused rating records its origin;
reuse is not an additional independent judgment. No held-out ratings are reused.

Primary selection is mean appropriateness with Standard and Local weighted
equally. Naturalness, style fit, simulated would-send rate and generation failures
are reported alongside it. Confidence intervals resample the 50 conversations
as paired blocks, keeping their styles and ten correlated persona scores
together (5,000 bootstrap samples, seed 425). A clear recommendation is withheld
when the top model's paired interval includes zero against a competitor.

The runner saves resumable per-model outputs, `progress.json`, the blinded packet,
an append-only `ratings.jsonl`, and, after complete matched coverage, `result.json`
and `report.md`. An incomplete run produces no model recommendation. Completion
does not automatically replace the V5 generator in the demo or revise historical
V6/V7 results. These are validation model-selection results from AI personas,
not independent human evidence or a new held-out performance claim.
