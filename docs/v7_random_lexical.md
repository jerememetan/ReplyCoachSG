# V7 v3: wording refinement of SEA-LION Local

On 9 November 2026 the user chose to refine SEA-LION Local directly. Standard
remains the original SEA-LION reply. The other three models remain comparison
sources and cannot replace either base reply.

## How it works

1. Generate the original Standard/Local pair from SEA-LION.
2. Find eligible whole phrases in the original Local text. Particle forms such
   as `can ah` are eligible wherever their source phrase matches.
3. Sample a single edit for up to three distinct phrases, then try
   two-phrase combinations and other single edits. Keep up to six distinct
   variants alongside the original.
4. If there is at least one variant, Qwen + DailyDialog ranks the original and
   the available variants using its existing conversation and style criteria.
   The original can win.

For example, `Please eat.` can yield `Pls eat.`, `Please makan.`, and
`Pls makan.`. `Okay, meet at 7.` can yield `Okay lor, meet at 7.`,
`Okay ah, meet at 7.`, or `Okay la, meet at 7.`. `Ok` has the same particle options.
`Can meet at 7?` can yield `Can ah meet at 7?`, `Can lah meet at 7?`, and
`Can lor meet at 7?`. Qwen ranks these candidates alongside the original.

Common chat phrases also have particle variants, such as `no worries lah`,
`let's go lah`, `take care ah`, and `not sure leh`.

The dictionary contains **160 conversion groups / 161 input phrases**, including
never mind/nevermind → nvm, tomorrow → tmr/tmrw, please → pls/plz, you → u,
remember → rmb, already → alr, eat → makan, invite → jio, sorry → paiseh, and
thank you/thanks → xiexie / xie xie as additional options. Abbreviations make
text casual and by themselves provide limited evidence of Singlish authenticity.

`scripts/update_chat_variations.py` reuses the existing augmentation
`abbreviation_snapshot()` allowlist from `reranker_packaging.py`, whose original
table is `scripts/data-split/near_dups.py`. It merges these forms into the
inference JSON and records the source hash; it does not run augmentation or
training. It also imports enabled eat/invite/sorry/do-not-have mappings from
the existing local-term lexicon, with its source hash. The source tables remain
unchanged. Rerun this script to refresh the merged dictionary, then restart the
demo to clear its cached dictionary.

The refresh script includes 15 replacement forms from Justin Peh's
English-to-Singlish preprocessing rules. `because → cos` was already present.
Following review of the 60-conversation replay, `you → u` is the only
replacement for `you`. Contractions support straight and curly
apostrophes and the source's unpunctuated spellings. V7 restricts `liddat` to
`do it like that` so it preserves the meaning of replies such as `I like that`.
Some source forms, including `cheena`, `engrish`, and `sinkie`, can read as mocking.
[Justin Peh's English-to-Singlish rules](https://github.com/justinpeh97/singlish-english-nmt/blob/main/data_generate/clean_english_to_singlish.txt)

The refresh script also reverses the spelling pairs from Peh's English cleanup
file. For example, `people` can become `ppl` or `pple`, and `should` can become
`shld` or `shd`. It deduplicates repeated pairs and skips entries that only
remove apostrophes. Common spellings of contractions share the same variants.
[Justin Peh's English cleanup rules](https://github.com/justinpeh97/singlish-english-nmt/blob/main/data_generate/clean_english_replace.txt)

Four phrase substitutions use senses documented in Wiktionary.

| Input phrase | Replacement | Source |
| --- | --- | --- |
| save me a seat | chope a seat for me | [chope](https://en.wiktionary.org/wiki/chope#English) |
| get the food to go | dabao the food | [tapao](https://en.wiktionary.org/wiki/tapao#English) |
| hang out after class | lepak after class | [lepak](https://en.wiktionary.org/wiki/lepak#English) |
| just make a rough guess | just agak-agak | [agak-agak](https://en.wiktionary.org/wiki/agak-agak#English) |

The whole normalization table is not blindly reversed. For example train → MRT
can introduce a location/transport assumption, nice → shiok is context-sensitive,
and disabled agreement mappings remain excluded. Treat is not mapped to jio:
jio means invite. The user's xiexie spellings are explicitly recorded additions.

Longest phrases match first; word boundaries prevent edits inside words.
Matching uses the original text so conversions do not cascade. URLs, emails,
inline/fenced code, punctuation and capitalization are protected/preserved.
The retrieval/normalization lexicon is separate and unchanged.

## Short replies and errors

For example, `Please!` has two permitted shortenings, so Qwen can rank those
two against the original. With zero variants,
the app keeps the original, shows the skipped explanation, and skips loading Qwen.

If dictionary loading or Qwen fails, keep the original Local reply, show the
error and allow retry. If SEA-LION itself fails, show its error and comparison
proposals without substituting another final-output model.

## Trace and evaluation boundary

The JSON export identifies `sealion_standard_lexical_v7_local`, with method
`lexical_phrase_diversity_v3`, dictionary hash, original/variants/selected
reply, Qwen identity/scores, timings, status and any skip/error. The UI's Local
wording candidates panel shows the same choice. Live sampling is unseeded;
tests/research callers can inject a seed for reproducibility. No new user setting,
fine-tuning, generator calls or Gemma criteria were introduced.

The old particle-oriented `lexicon_variations_v1` remains unchanged for historical
research/evaluation. The [isolated V7 v2 Local evaluation](evaluation/v7_v2_local_20261109.md)
reports the SEA-LION-anchored method on the same 60 revisited cases with one fixed
draw per conversation and the same Gemma rubric and personas. That run required
three distinct variants to rank. Its scores apply to that evaluated policy.
The [V7 v3 evaluation](evaluation/v7_v3_local_20261115.md) reports the
final expanded-particle replay, the intermediate standalone-only run, and the
separate development experiment with reviewed replies.
