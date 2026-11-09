# Six-system pipeline findings — 8 November 2026

**Main finding:** SEA-LION alone (V5) has the highest mean appropriateness in both styles. V7 v1 improves Local style fit and simulated would-send over V6. The evaluated cross-model Qwen selector scores lower on appropriateness than SEA-LION alone. Evaluation did not change model weights.

**Subsequent product decision (9 November):** after reviewing these results and
live failures, the user requested SEA-LION as the base for both styles, keeping
Standard direct and subsequently adding random lexical v2 refinement of Local
only. Qwen no longer selects across the four models. See the
[new method and evaluation boundary](../v7_random_lexical.md). This is a decision after test inspection,
not a newly validated pipeline or a fresh unseen-test result. Frozen runs and
the aggregate JSON below still describe the evaluated six-system configuration.

## Scope and verification

- Same 60 previously inspected test conversations; fixed reviewed 190-conversation reply bank; M2 memory. SEA-LION was selected earlier on the separate 50-conversation validation comparison.
- Six systems: BM25 (V0/V1 with normalization), MiniLM (V2), V4 attention RNN, V5 SEA-LION 3B, V6 Qwen + DailyDialog selection, and V7 Local lexicon refinement. V3 is excluded: supplied exports cover validation only and trained weights were not received for these test cases.
- 720 blinded candidate rows, both styles, ten fixed fictional Gemma4:12b personas: 7,200 complete cells, 600 per system/style; no failed cells. Same current rubric, temperature 0, seed 425, context 4096, prediction limit 2048, thinking disabled.
- 4,950 fresh Gemma calls and 2,250 verified exact conversation/style/reply/persona reuses within this run. No historical-rubric grades reused. Identical outputs receive identical grades, including V6/V7 Standard.
- Validated journal identity, all candidate/persona coverage, packet/freeze hashes, exact-reuse provenance, all summary metrics, and the saved paired intervals against raw evidence. All 60 SEA-LION outputs report CUDA execution.
- Cluster job 358114 on aloha, RTX 3090: successful exit 0:0 after 6h 45m 33s; finished 8 November at 22:26:48 Singapore time. Raw evidence is mirrored locally and remains ignored.

## All numerical grading criteria

Scores are /5. Would-send is simulated willingness with little or no editing, not human acceptance. Explanations, problem categories and uncertainty are also recorded; explanations are not an additional score.

| System | Style | Appropriateness | Naturalness | Style fit | Would send |
|---|---|---:|---:|---:|---:|
| BM25 | STANDARD | 2.787 | 4.898 | 5.000 | 38.7% |
| BM25 | LOCAL | 3.607 | 4.768 | 4.758 | 62.2% |
| MiniLM | STANDARD | 3.037 | 4.917 | 5.000 | 44.7% |
| MiniLM | LOCAL | 3.828 | 4.845 | 4.795 | 67.5% |
| V4 | STANDARD | 2.393 | 4.657 | 5.000 | 23.0% |
| V4 | LOCAL | 2.663 | 4.620 | 3.105 | 23.2% |
| V5 | STANDARD | 4.555 | 4.975 | 5.000 | 88.0% |
| V5 | LOCAL | 4.552 | 4.898 | 3.790 | 78.3% |
| V6 | STANDARD | 3.995 | 4.940 | 5.000 | 71.5% |
| V6 | LOCAL | 4.383 | 4.845 | 3.887 | 73.0% |
| V7 | STANDARD | 3.995 | 4.940 | 5.000 | 71.5% |
| V7 | LOCAL | 4.440 | 4.915 | 4.875 | 85.2% |

## Overall, with equal weights for the two styles

| System | Appropriateness | Naturalness | Style fit | Would send |
|---|---:|---:|---:|---:|
| BM25 | 3.197 | 4.833 | 4.879 | 50.4% |
| MiniLM | 3.433 | 4.881 | 4.897 | 56.1% |
| V4 | 2.528 | 4.638 | 4.053 | 23.1% |
| V5 | 4.553 | 4.937 | 4.395 | 83.2% |
| V6 | 4.189 | 4.893 | 4.443 | 72.2% |
| V7 | 4.218 | 4.928 | 4.938 | 78.3% |

## Paired uncertainty for pipeline changes

Resample 60 conversations as paired blocks, retaining all ten personas per conversation; 5,000 bootstrap samples, seed 425. These are descriptive, pairwise intervals, unadjusted for multiple comparisons. Would-send differences are percentage points. The V7-vs-V5 comparison is an added descriptive analysis; the saved run already contained V6-vs-V5 and V7-vs-V6.

| Change | Style | Metric | Difference | 95% paired interval |
|---|---|---|---:|---:|
| V6 − V5 | STANDARD | appropriateness | -0.560 | [-0.903, -0.258] |
| V6 − V5 | STANDARD | naturalness | -0.035 | [-0.083, +0.002] |
| V6 − V5 | STANDARD | style_fit | +0.000 | [+0.000, +0.000] |
| V6 − V5 | STANDARD | would_send | -16.500 pp | [-26.667, -7.667] pp |
| V6 − V5 | LOCAL | appropriateness | -0.168 | [-0.337, -0.040] |
| V6 − V5 | LOCAL | naturalness | -0.053 | [-0.110, -0.008] |
| V6 − V5 | LOCAL | style_fit | +0.097 | [-0.043, +0.262] |
| V6 − V5 | LOCAL | would_send | -5.333 pp | [-11.667, -0.167] pp |
| V7 − V6 | STANDARD | appropriateness | +0.000 | [+0.000, +0.000] |
| V7 − V6 | STANDARD | naturalness | +0.000 | [+0.000, +0.000] |
| V7 − V6 | STANDARD | style_fit | +0.000 | [+0.000, +0.000] |
| V7 − V6 | STANDARD | would_send | +0.000 pp | [+0.000, +0.000] pp |
| V7 − V6 | LOCAL | appropriateness | +0.057 | [+0.017, +0.107] |
| V7 − V6 | LOCAL | naturalness | +0.070 | [+0.015, +0.132] |
| V7 − V6 | LOCAL | style_fit | +0.988 | [+0.708, +1.262] |
| V7 − V6 | LOCAL | would_send | +12.167 pp | [+5.833, +19.167] pp |
| V7 − V5 | STANDARD | appropriateness | -0.560 | [-0.903, -0.258] |
| V7 − V5 | STANDARD | naturalness | -0.035 | [-0.083, +0.002] |
| V7 − V5 | STANDARD | style_fit | +0.000 | [+0.000, +0.000] |
| V7 − V5 | STANDARD | would_send | -16.500 pp | [-26.667, -7.667] pp |
| V7 − V5 | LOCAL | appropriateness | -0.112 | [-0.270, +0.013] |
| V7 − V5 | LOCAL | naturalness | +0.017 | [-0.050, +0.087] |
| V7 − V5 | LOCAL | style_fit | +1.085 | [+0.817, +1.352] |
| V7 − V5 | LOCAL | would_send | +6.833 pp | [-0.667, +14.500] pp |

## Interpretation

- V6 versus SEA-LION alone: appropriateness declines by 0.560 Standard and 0.168 Local; both paired intervals exclude zero. Current selection is a measured bottleneck on this set. This comparison does not isolate whether training data, objective, candidate mix or domain shift caused it.
- V7 versus V6 Local: appropriateness +0.057, naturalness +0.070, style fit +0.988 and would-send +12.17 percentage points; all four paired intervals exclude zero in this descriptive analysis. Standard is unchanged.
- V5 remains the appropriateness leader: 4.555 Standard / 4.552 Local, compared with V7 at 3.995 / 4.440. V7 has the highest Local style fit (4.875) and would-send point estimate (85.2%). Do not call V7 universally best; compare the added V7-vs-V5 intervals above.
- High grammar/naturalness scores can coexist with low appropriateness (for example V4). A fluent, style-compliant reply can still miss the conversational intent.

## Problem and uncertainty flags

Counts are out of 600 cells per row. Generic/unrelated is the judge label, not evidence of a failed generator. All problem categories, score distributions and derived appropriateness ≤2 counts are in the companion JSON.

| System | Style | Generic/unrelated | Other problem flags | Uncertain | Appropriateness ≤2 |
|---|---|---:|---:|---:|---:|
| BM25 | LOCAL | 79 | 0 | 5 | 219 |
| BM25 | STANDARD | 156 | 0 | 0 | 341 |
| MiniLM | LOCAL | 75 | 0 | 1 | 182 |
| MiniLM | STANDARD | 152 | 0 | 0 | 283 |
| V4 | LOCAL | 191 | 1 | 0 | 332 |
| V4 | STANDARD | 239 | 0 | 3 | 395 |
| V5 | LOCAL | 10 | 0 | 0 | 66 |
| V5 | STANDARD | 1 | 0 | 0 | 65 |
| V6 | LOCAL | 11 | 0 | 0 | 89 |
| V6 | STANDARD | 31 | 0 | 0 | 148 |
| V7 | LOCAL | 10 | 0 | 0 | 83 |
| V7 | STANDARD | 31 | 0 | 0 | 148 |

## Selection and inference overhead

V6 selected proposal sources (ties/deduplicated multi-source replies are reported together):
- STANDARD: `{"BM25": 4, "BM25+MiniLM": 5, "MiniLM": 8, "V4": 1, "V5": 42}`.
- LOCAL: `{"BM25": 4, "BM25+MiniLM": 2, "MiniLM": 4, "V4": 2, "V5": 48}`.
- V7 changed 50/60 Local replies and retained 10/60 originals; Standard stayed identical in all 60 cases.
- Mean V7 overhead on this RTX 3090 batch: lexicon sampling 0.48 ms, reranking 0.079 s. These are batch-run measurements, not a new laptop Streamlit benchmark.
- V7 creates exactly three distinct Local lexicon variants plus the original, then uses the existing ranker. Per-case seed `425:<item_id>` freezes this randomized evaluation. No extra generator or Gemma call occurs during demo inference.

## Limits and presentation guidance

- These 60 test conversations were already inspected in earlier experiments; this is a re-evaluation after validation-based SEA-LION selection, not a fresh unseen test or model-selection set. No tuning from these results.
- Ten personas share one Gemma model and are correlated; they are not ten independent human judges. Bootstrap conversations, not individual persona ratings. No human authenticity or usability improvement is established.
- Singlish particles remain context-sensitive; the earlier “sia” diagnostic remains a valid failure case. Aggregate Local style gains do not demonstrate that every added particle fits meaning or tone.
- The comparison fixes M2, reply bank, providers and ranker; it does not separate architectural effects from data, checkpoint or decoding choices. Do not mix these scores with historical mT5/older-rubric scores.
- SEA-LION validation selection remains a separate comparison. The subsequent user-requested direct demo path is a product decision after inspecting this report; it does not change the evaluated system definitions or establish fresh unseen-test performance.

## Evidence and reproduction

- Entry point: `scripts/run_final_test.py --current-pipeline --out experiments/judging/<new-run>` on an allocated CUDA GPU with the pinned models and Gemma available. Use `--resume` only for identical interrupted inputs.
- Private run: `experiments/judging/final_current_20261108_01/` (`freeze.json`, `outputs.jsonl`, anonymous packet/private mapping, `judge_run.json`, `ratings.jsonl`, `result.json`, `report.md`).
- Public [aggregate evidence](current_pipeline_20261108.json) contains hashes, all metrics/distributions, paired intervals, execution and reuse counts; excludes conversations, reply text, persona explanations, credentials and weights.
- Historical final-test and transformer-selection evidence remains unchanged.
