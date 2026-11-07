# Frozen mT5 linear probing

Job 357219 completed successfully on an L40S in 30 seconds, after adapter job 357048. All 4,608 saved linear-probe predictions were checked against the cached representations and nine saved classifier heads. Frozen checkpoint hashes remained unchanged. The nine selected heads and raw report are also saved locally in `artifacts/mt5_linear_probe_v1/`.

We compare three frozen checkpoints: the original fully adapted mT5 base, the latest full English fine-tune (356868), and the English LoRA (357048). The probe does not update any model or adapter weights.

For each of availability, completion and ownership, the diagnostic has 384 training, 64 validation and 64 reserved development examples. Conversations state both opposing facts. Query targets, fact order and entity-to-label associations are counterbalanced. Opposite-label association swaps have exactly the same word counts. Template families and concrete entity strings are disjoint between these diagnostic splits.

The classifier is logistic regression on one fixed encoder representation, with 768 coefficients and an intercept. Validation selects representation and regularization. Controls include a word-unigram classifier, uncontextualized embeddings and shuffled training labels. Scores include per-example accuracy and whether both members of each query or association pair are correct.

These are assistant-authored synthetic diagnostics, with correlated paired examples. They are not project test accuracy or independent human ratings. A successful probe establishes that information is accessible to this readout on this diagnostic. It does not establish that generation uses the information. An unsuccessful probe does not establish that the information is absent.

Recipe: `mt5_linear_probe_v1_recipe.json`. Launch record: `mt5_linear_probe_v1_launch.json`. Dataset: `../data/diagnostics/mt5_linear_probe_v1/rows.json`.

## Results

Each task has 64 reserved synthetic development examples; random guessing is 50%. These paired examples are correlated, not 64 independent real conversations.

| Frozen checkpoint | Availability | Completion | Ownership |
|---|---:|---:|---:|
| Adapted public-data base | 51.6% | 54.7% | 54.7% |
| Full English fine-tune | 51.6% | 53.1% | 62.5% |
| English LoRA | 50.0% | 53.1% | 59.4% |
| Word-unigram control | 50.0% | 50.0% | 50.0% |

Shuffled-label controls score 46.9–53.1%. The strongest ownership probe (full English) gets both opposite-label fact-binding versions right for 8/32 pairs. English-LoRA completion fits 99.7% of training examples but reaches only53.1% on reserved examples. Training fit does not establish robust transfer.

Reviewed all24 paired generated replies per checkpoint (72outputs). Examples still reverse completion, confuse which object was asked about, reverse return ownership, and invent meeting times. These are qualitative observations, not independently rated generator accuracy.

The pilot does not establish a reliable linear-probe improvement. Information may still be encoded nonlinearly or lost by these pooling choices; low scores do not prove that the model lacks the facts. No probe has been added to the reply pipeline and no model default has changed.

Data-quality note: one generic completion question uses singular “Is” with plural “welcome kits.” Factual labels and counterbalancing remain valid. Frozen evaluated inputs are retained rather than silently changed.

Verified summary: `mt5_linear_probe_v1_results.json`. Raw scores, selected heads, all candidate validation scores and predictions are in `../artifacts/mt5_linear_probe_v1/`.
