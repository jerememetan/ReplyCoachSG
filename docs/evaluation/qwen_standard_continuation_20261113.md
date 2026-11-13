# Standard V6 continuation and SEA-LION fallback

We continued the [fine-tuned V6 adapter](qwen_refinement_20261113.md) for two rounds to see whether more supervision would bring Standard simulated would-send to 80% on the presentation replay. Each conversation had a fixed candidate pool from BM25, MiniLM, V4 and SEA-LION. The reranker scored those candidates with M2 memory.

## Training and evaluation

We split 256 authored synthetic conversations by template before generation. Each training round received 77 conversations from its own families. Another 39 conversations were allocated to validation and 63 to test. Jev compared SEA-LION with up to two alternatives favored by the current checkpoint. A separate Gemma pass graded a subset of unresolved comparisons, prioritizing close scores or switches away from SEA-LION.

| Round | New preferences | Jev | Gemma | Disagreed with incoming checkpoint or tied | Training reply rows | Optimizer steps |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 66 | 58 | 8 | 4 | 656 | 82 |
| 2 | 54 | 50 | 4 | 2 | 764 | 96 |

Both rounds used two epochs at a learning rate of 7.5e-6. Round 2 continued the first round's adapter and trained with both rounds' labels. The earlier study's validation loss selected epoch 2 in each round. The new validation set compared the adapters at fixed score margins for switching away from SEA-LION. Direct SEA-LION won overall and original V6 with no margin was the best reranking policy. The held-out evaluation used 210 unique blinded Gemma grades across three simulated personas.

## Results

| Standard policy | Presentation appropriateness / 5 | Presentation naturalness / 5 | Presentation would send | Fresh appropriateness / 5 | Fresh would send |
|---|---:|---:|---:|---:|---:|
| Original V6 | 3.995 | 4.940 | 71.5% | 3.069 | 46.0% |
| Previous fine-tuned V6 | 4.135 | 4.922 | 75.2% | 3.021 | 44.4% |
| Continuation round 1 | 4.135 | 4.922 | 75.2% | 3.069 | 46.0% |
| Continuation round 2 | 4.118 | 4.922 | 75.0% | 3.069 | 46.0% |
| Direct SEA-LION | 4.555 | 4.975 | 88.0% | 3.201 | 50.8% |

Style fit was 5.000 for every policy on both evaluations. Fresh-test naturalness was also 5.000 for every policy. Each continued adapter switched away from SEA-LION on six of 63 fresh conversations.

Neither continuation reached the presentation set's 80% target. The first tied the previous fine-tuned adapter on that replay, and the second slipped slightly. Both recovered to original V6 on the fresh test. Direct SEA-LION led both evaluations. The 120 added preferences included only six cases where the checkpoint used for mining favored the rejected reply or tied it, limiting what this run says about adding more examples of actual ranking errors. The presentation replay reused saved grades, and the fresh test used six synthetic scenario families. Would-send percentages describe Gemma personas' judgments. The demo retains direct SEA-LION for Standard.
