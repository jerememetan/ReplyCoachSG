# Direct SEA-LION demo selection

Historical plan: the later [random Local wording plan](2026-11-09-random-lexical-local.md)
adds V7 v2 to the SEA-LION Local reply while keeping Standard direct.

The user approved SEA-LION as the product path after reviewing the completed
comparison and live reranker failures. All models (experimental) continues to
show BM25, MiniLM, V4 and SEA-LION proposals, but its two final replies come
directly from SEA-LION. No V6 model or V7 lexicon pass loads in this mode.

Implementation and validation:

- Add a separate direct-comparison runner/result; retain the historical
  `run_all_models` and `run_v7_all_models` implementations unchanged.
- Preserve existing retrieval/RNN proposal behavior. Keep SEA-LION's original
  Standard/Local text without a heuristic or learned selection pass.
- Remove the shared-selector control from this demo mode and render clear
  SEA-LION provenance, the comparison table, clickable replies and JSON export.
- If SEA-LION fails, show its error with the remaining comparison proposals;
  do not invent final replies or silently use another model. Retry stays usable.
- First demonstrate failure with focused runner/UI tests, then implement and
  run the demo, resource, V6/V7 and evaluation regression checks.
- Update current demo documentation while preserving historical findings.

The existing overall entry mode, other demo modes, private weights, datasets,
frozen results, judge rubric and historical selection runners stay unchanged.
This is an explicit user product decision after inspecting test results, not
an independently validated new pipeline or a new claim of unseen-test quality.
