# Random lexical refinement of SEA-LION Local

The user approved per-phrase random shortening and explicitly chose refining
SEA-LION Local directly. Standard remains the original SEA-LION reply. Qwen
never selects across BM25/MiniLM/V4/SEA-LION in the product path.

- Use a separate phrase/abbreviation dictionary, preserving the retrieval
  lexicon and historical V7 v1 implementation and evaluation artifacts.
- Each eligible non-overlapping phrase occurrence gets an independent 50% roll
  per attempt; on conversion, select one allowed replacement. Match the longest
  phrase, protect URLs/emails/code, preserve punctuation and casing, and never
  cascade replacements within an attempt.
- Generate three distinct changed candidates with bounded retries, retaining
  the original as the fourth. Discard unchanged/duplicate attempts. If three
  cannot be produced, retain the original with a visible skipped explanation;
  do not force particles or call Qwen for an undersized pool.
- Qwen scores exactly those four Local candidates using the existing criteria.
  Record a new method identity, edits, scores, timings, and original/final text.
  On a ranker error, keep SEA-LION Local, show the error and permit retry.
- Add focused tests first, implement, run legacy and current regression checks,
  then verify the real Streamlit path. No configurable probability, training,
  new judge criteria, or reuse of v1 quality claims for v2.

The 50% probability applies before rejecting duplicate/unchanged attempts, not
to a promise that half of the words in each final selected reply will change.
