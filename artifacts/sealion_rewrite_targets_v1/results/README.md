# SEA-LION English-to-Singlish target wording trial

This experiment continues the original repair25 Singlish LoRA on fixed English drafts. It does not train English generation. The unchanged original is evaluated beside two matched continued-training runs: old targets and revised targets.

The 745 existing rewrite pairs are split into 625 replay pairs and a 120-pair expansion-only overlay. The revised arm changes 100 targets; 20 already-natural targets are unchanged. No team-authored source record or original data file is edited. Both runs also receive 20 new everyday pairs, so improvement over the original alone cannot isolate the target revision.

Each run has 240 updates with 5 replay, 1 shared everyday and 2 overlay examples per update. Every overlay example appears four times. Both start from the same original rank-8 Singlish adapter; the base model and English adapter stay frozen. Learning rate is 1e-5, batch 8, reply-only cross entropy. No copy penalty, lexicon substitution or reranking is used.

Snapshots 0/60/120/240 retain all adapters and generate the same 20 new validation cases. Lowest validation reference CE selects a candidate including step zero; final 240 remains separately available. Selection loss does not establish naturalness. All candidates require separate review for meaning, tone and natural Singlish.

Selected and final candidates are compared with the original on 20 new illustrative test cases, 10 fidelity controls, 12 saved actual English drafts from six previous casual conversations, and 42 reused old validation cases. English is identical across routes, including any existing English mistakes. The new targets and cases are assistant-authored preferences, not independently human-rated gold or benchmark accuracy.

`style_preview.csv` shows 15 target examples, not model-generated results. `recipe.json` records all hashes and training details. `launch.json` will record personal-account L40S jobs. The summary job collects CSV/JSON comparisons and a results-only archive after both runs finish. Original model files and the independent protected backup are never training outputs. This trial does not promote a default model.
