# Role B: Retrieval, Memory and Score-Calibration Investigation

## 1. Objective

Compare BM25 and MiniLM retrieval across memory configurations M0, M1 and M2 using the same 50 development validation queries and the disjoint 190-item training/reply bank. Investigate retrieval quality, latency, fallback behaviour and the comparability of retrieval scores before changing the shared reranker.

## 2. Experiments performed

Six retrieval-only exports were generated:

- BM25: M0, M1 and M2
- MiniLM: M0, M1 and M2

Each configuration produced 50 validation-query records. The exports were analysed for output differences, fallback counts, latency and score distributions.

## 3. Latency and fallback results

| Retriever | Memory | Mean latency | P95 latency | STANDARD fallbacks | LOCAL fallbacks |
|---|---|---:|---:|---:|---:|
| BM25 | M0 | 5.36 ms | 9.84 ms | 0 | 0 |
| BM25 | M1 | 4.82 ms | 6.71 ms | 0 | 0 |
| BM25 | M2 | 4.70 ms | 6.51 ms | 0 | 0 |
| MiniLM | M0 | 18.57 ms | 27.55 ms | 1 | 1 |
| MiniLM | M1 | 19.57 ms | 25.53 ms | 0 | 0 |
| MiniLM | M2 | 19.96 ms | 25.78 ms | 0 | 0 |

Latency was measured locally with the retriever initialised before timing and one warm-up per configuration. Index initialisation was excluded. These figures are local benchmark results, not production latency guarantees.

MiniLM M0 had one affected query, ITEM-D-009, which produced both a STANDARD and a LOCAL fallback. The incoming message concerned a Gaussian elimination question, but the fallback responses asked about confirming a time and place. Both fallback scores were zero, and neither response had a retrieved item ID.

## 4. Retrieval and memory comparison

The existing configuration comparison reported:

- 50/50 validation queries had a BM25–MiniLM retrieval disagreement.
- 48/50 queries had a reply change when memory configuration changed.
- Only 1/50 query had a difference between M1 and M2.
- No export errors were recorded.

These figures demonstrate that the configurations often select or produce different outputs. They do not establish which output is more relevant or appropriate.

Preliminary manual relevance scoring covered only 24 of the 50 queries. Its results are incomplete and should not be treated as a definitive quality ranking. The existing five-conversation judge pilot compared BM25 M0 and M2 only; it did not compare MiniLM.

## 5. Score-scale investigation

The shared reranker currently calculates:

`final_score = 0.6 × input_score + 0.4 × token_overlap`

BM25 input scores are normalised relative to the highest-ranked result for each query. Consequently, their scores cluster near 1.0.

Across the 50 exported candidates per style, mean input scores were:

| Configuration | STANDARD mean | LOCAL mean |
|---|---:|---:|
| BM25 M0 | 0.9964 | 0.9918 |
| BM25 M1 | 0.9945 | 0.9981 |
| BM25 M2 | 0.9945 | 0.9981 |
| MiniLM M0 | 0.4728* | 0.4706* |
| MiniLM M1 | 0.5360 | 0.5343 |
| MiniLM M2 | 0.5348 | 0.5332 |

*MiniLM M0 had 49 ordinary input scores per style; one fallback per style lacked the input-score metadata.*

BM25 and MiniLM therefore have substantially different input-score distributions. The fact that their scores are bounded within [0, 1] does not establish that they are calibrated or directly comparable.

The BM25 normalisation also means a score near 1.0 indicates a candidate near the top of that query's ranking, not necessarily a highly relevant candidate. MiniLM's lower cosine-similarity scores do not, by themselves, indicate worse retrieval quality.

## 6. Limitations

1. The comparison covers 50 validation queries and does not establish generalisation to unseen data.
2. Manual relevance scoring is available for only 24 queries and is preliminary.
3. The existing judge pilot used five conversations and compared BM25 M0 and M2, not BM25 against MiniLM.
4. The judge is a local LLM evaluator, not an independent human gold standard. The pilot showed style-fit ceiling effects and contextual mistakes.
5. Latency figures are local measurements and exclude retriever initialisation.
6. Output disagreement and raw score magnitude are not measures of accuracy.
7. The MiniLM M0 fallback should be analysed separately from ordinary retrieval candidates.
8. The current results do not establish an optimal reranker weight or demonstrate that a hybrid method outperforms either retriever.

## 7. Proposed next actions

1. Ask C/D to review the score-scale interpretation and agree on a fair relevance-based comparison.
2. Evaluate candidate relevance and ranking behaviour on the same development queries, treating fallback cases separately.
3. If a scaling or weight change is proposed, compare it against the existing reranker on development data using an agreed evaluation procedure.
4. Keep the current reranker unchanged until the evidence supports a modification.
5. Do not tune on the 60 locked test items.

**Current conclusion:** The experiments establish meaningful differences in score distributions, latency and selected outputs. They do not yet establish that BM25 or MiniLM is superior, that their scores are calibrated, or that changing the reranker improves reply quality.
