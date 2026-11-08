# Transformer selection findings — 8 November 2026

**Decision:** use SEA-LION 3B for the interactive All models V5 slot. It leads on the predeclared primary criterion: mean appropriateness, with Standard and Local weighted equally.

## Scope and method

- Original V5: mT5 + LoRA (`artifacts/spk1_ul025_s425/best`).
- New mT5 A: shared-intent rewrite, seed 1, final784 handoff.
- SEA-LION 3B: English356177 + Singlish repair357651, step25 handoff.
- Same 50 reviewed validation conversations, M2 memory and inferred generator speakers; no retrieval, Qwen selection or V7 refinement in this comparison.
- 300 blinded candidates: 3 models × 50 conversations × 2 styles. Ten existing fictional personas give 3,000 complete rating cells (500 per model/style).
- Existing Gemma4 12B model, rubric and criteria; temperature 0, seed 425, context 4096, prediction limit 2048, thinking disabled. No new grading criteria or training.
- 2,960 actual fresh judgments and 40 verified exact-task/persona reuses. No older validation ratings were reused because their saved rubric differed.
- 141 laptop ratings were retained; 2,819 fresh judgments ran on cluster L40S job 358003. The parallel laptop journal was backed up; overlapping work was not double-counted.
- All three models generated all 50 reply pairs; zero generation failures and zero failed rating cells. Held-out test data was excluded from selection.

## Every scored criterion, by style

The three scores are on 1–5 scales. Would-send is simulated willingness with little or no editing, not actual user acceptance.

Every rating also contains a short explanation. These rationales remain in the private raw journal; they are supporting text, not an additional numeric score.

| Model | Style | Appropriateness | Naturalness | Style fit | Would send |
|---|---|---:|---:|---:|---:|
| Original mT5 | STANDARD | 3.804 | 4.948 | 5.000 | 66.4% |
| Original mT5 | LOCAL | 4.270 | 4.954 | 3.896 | 74.6% |
| New mT5 A | STANDARD | 3.554 | 4.944 | 5.000 | 57.4% |
| New mT5 A | LOCAL | 3.748 | 4.638 | 4.160 | 59.2% |
| SEA-LION 3B | STANDARD | 4.486 | 4.958 | 5.000 | 83.2% |
| SEA-LION 3B | LOCAL | 4.546 | 4.964 | 4.206 | 83.4% |

## Overall (equal style weights)

| Model | Appropriateness | Naturalness | Style fit | Would send |
|---|---:|---:|---:|---:|
| Original mT5 | 4.037 | 4.951 | 4.448 | 70.5% |
| New mT5 A | 3.651 | 4.791 | 4.580 | 58.3% |
| SEA-LION 3B | 4.516 | 4.961 | 4.603 | 83.3% |

## Paired uncertainty for all metrics

SEA-LION minus each competitor. Resample 50 conversations as paired blocks, retaining both styles and all ten personas together; 5,000 samples, seed 425. Intervals are descriptive, pairwise and unadjusted for multiple comparisons. Proportion differences below are percentage points; lower is better for uncertainty/problem flags.

| Competitor | Metric | Difference | 95% bootstrap interval |
|---|---|---:|---:|
| Original mT5 | appropriateness | +0.479 | [+0.109, +0.856] |
| Original mT5 | naturalness | +0.010 | [-0.037, +0.052] |
| Original mT5 | style_fit | +0.155 | [-0.021, +0.335] |
| Original mT5 | would_send | +12.800 pp | [+1.800, +24.200] pp |
| Original mT5 | uncertain | +0.100 pp | [+0.000, +0.300] pp |
| Original mT5 | problem_flagged | -7.200 pp | [-13.300, -1.600] pp |
| New mT5 A | appropriateness | +0.865 | [+0.456, +1.296] |
| New mT5 A | naturalness | +0.170 | [+0.073, +0.279] |
| New mT5 A | style_fit | +0.023 | [-0.186, +0.235] |
| New mT5 A | would_send | +25.000 pp | [+13.100, +37.000] pp |
| New mT5 A | uncertain | +0.100 pp | [+0.000, +0.300] pp |
| New mT5 A | problem_flagged | -6.600 pp | [-10.800, -2.900] pp |

## Problem and uncertainty flags

Counts are out of 500 rating cells per model/style. `generic_or_unrelated_fallback` is the judge’s label; no generator failed or invoked a replacement generator in this run.

| Model | Style | Generic/unrelated flag | Uncertain | Appropriateness ≤2 |
|---|---|---:|---:|---:|
| Original mT5 | STANDARD | 57 | 0 | 149 |
| Original mT5 | LOCAL | 31 | 0 | 79 |
| New mT5 A | STANDARD | 55 | 0 | 188 |
| New mT5 A | LOCAL | 27 | 0 | 150 |
| SEA-LION 3B | STANDARD | 10 | 0 | 63 |
| SEA-LION 3B | LOCAL | 6 | 1 | 51 |

All models had zero `garbled_or_unreadable`, `offensive_or_unsafe`, and `personal_info` flags. The aggregate JSON includes every problem category, including `none`, and complete 1–5 score distributions. These observed flags do not establish general safety or judge reliability.

## Interpretation

- SEA-LION improves overall appropriateness by +0.479 versus original mT5 and +0.865 versus new mT5 A. Both paired intervals stay above zero.
- Simulated would-send increases from 70.5% (original) and 58.3% (new mT5 A) to 83.3% (SEA-LION).
- Naturalness is nearly tied with original mT5 (4.961 versus 4.951); do not describe this tiny point difference as a demonstrated gain.
- Standard style fit is 5.000 for every model, so it does not distinguish them. Local style-fit means are 3.896 original, 4.160 new mT5 A, and 4.206 SEA-LION; consult paired intervals before claiming style superiority.
- New mT5 A raises Local style fit relative to original mT5 but lowers appropriateness and would-send in both styles. More recognisable local phrasing did not imply a better conversational answer.
- The rubric explicitly does not establish Standard/Local meaning equivalence from separate single-reply grades; no paired semantic-equivalence claim is made.

## Runtime and deployment boundary

Recorded mean generator call times on the laptop were 0.804 s original, 1.665 s new mT5 A, and 3.833 s SEA-LION. These are diagnostics, not a controlled benchmark: handoff startup is counted inside the first call, while original V5 initialization occurs outside it. The cluster job completed in 2:30:43 including startup/download; an early 92-second parallel sample measured about 18.3 versus 9.8 ratings/minute.

The current Streamlit All models V5 slot uses SEA-LION. The frozen final-test entry points continue explicitly loading original mT5. Historical V6/V7 results must not be relabelled as SEA-LION results. A live pipeline smoke check establishes integration, not new end-to-end quality evidence.

## Limitations and evidence

- Correlated fictional AI personas, not independent human evidence. Validation model selection, not a held-out test.
- Ten personas are correlated views of one model.
- 141 ratings originated on the laptop; remaining fresh ratings used the cluster. Same model digest and grading settings; cross-runtime replay was not performed.
- Forty exact-task rating reuses are not independent additional inferences.
- Generator-only selection: SEA-LION plus Qwen/V7 has not received a new matched quality evaluation.
- Pairwise 95% bootstrap intervals are unadjusted for multiple comparisons.

The data is validation data used for model selection, and the candidates already have development histories. The 3,000 cells are not 3,000 independent conversations or human participants. Zero uncertainty flags in most groups do not prove the judge was certain or correct.

Private raw evidence: `experiments/judging/transformer_selection_20261108_01/` (ignored). The adjacent aggregate JSON records input/result hashes, judge identity, reuse counts and full metric distributions without publishing conversations or credentials.

Rebuild from the private evidence:

```powershell
.\.venv\Scripts\python.exe scripts/report_transformer_findings.py --run experiments/judging/transformer_selection_20261108_01 --out docs/evaluation/transformer_selection_20261108
```
