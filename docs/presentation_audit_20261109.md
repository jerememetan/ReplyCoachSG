# Presentation accuracy check — 9 November 2026

- **V0–V2:** TF-IDF and BM25 use lexical overlap; V1 keeps the original
  query tokens and adds canonical forms from the normalization lexicon; V2
  uses MiniLM embeddings and cosine similarity. All three retrieve stored
  replies. The [retrieval calibration](role_b_retrieval_calibration.md) used
  50 validation conversations and a 190-conversation bank. It does not
  establish that MiniLM or M2 gives better replies overall.
- **V6:** The historical setup ranked eight proposals, four per style. The
  [matched reranker comparison](reports/dailydialog_reranker_comparison_20261029.md)
  used the original mT5 on 50 validation conversations. The later
  [60-case regrade](evaluation/current_pipeline_20261108.md) used SEA-LION
  and is a separate comparison. SEA-LION alone led mean appropriateness.
- **V7:** The 60-case results measure particle-oriented V7 v1. Local style
  fit and simulated would-send improved over V6. The current
  [V7 v2 phrase refinement](v7_random_lexical.md) has been checked in the demo
  but has no new quality scores. Gemma persona ratings are simulated judgments.

For the demo, choose **All models (experimental)**, **M2**, and **Reviewed
development file** (`data/dev/train.jsonl`). Standard comes directly from
SEA-LION 3B. For Local, each eligible phrase in SEA-LION's reply gets a 50%
chance of conversion. Qwen + DailyDialog ranks the original plus three distinct
variants if available; otherwise the demo keeps the original. BM25, MiniLM, and
V4 remain visible as comparisons.
