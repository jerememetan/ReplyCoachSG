# V7 v2: random wording refinement of SEA-LION Local

On 9 November 2026 the user chose to refine SEA-LION Local directly. Standard
remains the original SEA-LION reply. The other three models remain comparison
sources and cannot replace either base reply.

## How it works

1. Generate the original Standard/Local pair from SEA-LION.
2. For each eligible whole phrase in the original Local text, independently roll
   a fixed 50% chance of conversion. If converting, randomly pick an allowed
   replacement from `data/lexicons/chat_variations.json`.
3. Sample three distinct changed replies; discard unchanged/duplicate attempts.
   The original is retained as the fourth candidate.
4. The existing Qwen + DailyDialog ranker scores only these four Local candidates
   with its existing conversation/style criteria. The original can win.

For example, `Never mind, please meet tomorrow.` can yield `Nvm, please meet
tmr.`, `Never mind, pls meet tomorrow.`, or combinations of these edits.
Every occurrence rolls separately, including repeated words. 50% applies before
rejecting duplicate/unchanged attempts; it does not mean exactly half the final
reply changes, or that the reranker chooses edited candidates half the time.

The dictionary contains **37 conversion groups / 38 input phrases**, including
never mind/nevermind → nvm, tomorrow → tmr/tmrw, please → pls/plz, you → u,
remember → rmb, already → alr, eat → makan, invite → jio, sorry → paiseh, and
thank you/thanks → xiexie / xie xie as additional options. Abbreviations make
text casual; they are not alone evidence of Singlish authenticity. No particles
are automatically appended.

`scripts/update_chat_variations.py` reuses the existing augmentation
`abbreviation_snapshot()` allowlist from `reranker_packaging.py`, whose original
table is `scripts/data-split/near_dups.py`. It merges these forms into the
inference JSON and records the source hash; it does not run augmentation or
training. It also imports enabled eat/invite/sorry/do-not-have mappings from
the existing local-term lexicon, with its source hash. The source tables remain
unchanged. Rerun this script to refresh the merged dictionary, then restart the
demo to clear its cached dictionary.

The whole normalization table is not blindly reversed. For example train → MRT
can introduce a location/transport assumption, nice → shiok is context-sensitive,
and disabled agreement mappings remain excluded. Treat is not mapped to jio:
jio means invite. The user's xiexie spellings are explicitly recorded additions.

Longest phrases match first; word boundaries prevent edits inside words.
Matching uses the original text so conversions do not cascade. URLs, emails,
inline/fenced code, punctuation and capitalization are protected/preserved.
The retrieval/normalization lexicon is separate and unchanged.

## Short replies and errors

Sampling is bounded to 64 attempts. If three distinct changed candidates cannot
be produced (for example `Please!` has only two permitted shortenings), keep the
original and show a skipped explanation. Do not force slang, pad duplicate
candidates, or load Qwen for an undersized candidate pool.

If dictionary loading or Qwen fails, keep the original Local reply, show the
error and allow retry. If SEA-LION itself fails, show its error and comparison
proposals without substituting another final-output model.

## Trace and evaluation boundary

The JSON export identifies `sealion_standard_lexical_v7_local`, with method
`lexical_bernoulli_v2`, probability, dictionary hash, original/variants/selected
reply, Qwen identity/scores, timings, status and any skip/error. The UI's Local
wording candidates panel shows the same choice. Live sampling is unseeded;
tests/research callers can inject a seed for reproducibility. No new user setting,
fine-tuning, generator calls or Gemma criteria were introduced.

The old particle-oriented `lexicon_variations_v1` remains unchanged for historical
research/evaluation. Its 8 November quality figures do **not** measure this new
SEA-LION-anchored v2 policy. No quality improvement or new held-out score is
claimed until v2 is evaluated separately.
