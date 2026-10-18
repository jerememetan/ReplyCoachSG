# Architecture

```
session (last 8 turns)
        |
        v
 +--------------+     +------------------+
 | M2 memory    | --> | V1 normalise     |
 | newest 2 +   |     | lexicon + CJK    |
 | 2 relevant   |     +------------------+
 +--------------+              |
                               v
                      +------------------+
                      | V0/V2 retrieve   |
                      | vetted pairs     |
                      +------------------+
                               |
                               v
                      +------------------+
                      | V4 or V5 generate|
                      | keep the winner  |
                      +------------------+
                               |
                               v
                      +------------------+
                      | V6 rerank        |
                      | + fallback       |
                      +------------------+
                               |
                STANDARD reply + LOCAL reply
```

## Shared candidate format

Every retriever, generator, and the reranker emits:

```json
{
  "context": ["yeah I'm starving"],
  "message": "eh later go makan?",
  "candidate_text": "Want to grab lunch first?",
  "style": "STANDARD",
  "source": "V0",
  "score": 4.21,
  "item_id": "PILOT-001",
  "normalised_message": "eh later go eat"
}
```

`style` is only `STANDARD` or `LOCAL`. `source` is one of `V0 V1 V2 V3 V4 V5 V6 fallback gold`.

## Experiment journey

| Version | What | Week |
| --- | --- | --- |
| V0 | TF-IDF / BM25 retrieval | 4 |
| V1 | Local-term detection and normalisation | 5 |
| V2 | Static embedding retrieval | 6 |
| V3 | RNN encoder-decoder | 5 |
| V4 | V3 + attention | 6–7 |
| V5 | mT5-small local fine-tune, compare with V4 | 6–7 |
| V6 | Context-candidate reranker | 6–8 |
| M0/M1/M2 | No memory / last 3 / dynamic | 6 and 8 |

## Fallback

If the best score for a style is below 0.15, emit a safe English line. LOCAL fallback is only slightly warmer English, never fake slang.

## Offline evaluation boundary

The persona-based LLM evaluation is separate from the reply-generation
pipeline above. An initial panel of ten AI personas rates anonymised saved
outputs on appropriateness, naturalness and style fit, each 1–5. Personas do not
generate user-facing replies, modify the authored dataset or supply training labels.
They need no tools or repository access. E owns `scripts/judge_replies.py`, which
reads `.env`/CSV, calls the standard local Gemma4 12B judge through Ollama sequentially,
and saves ratings and summaries. No hosted judge, API key, repository access or
private system mapping is sent to the model. Thinking is disabled, with a 4,096-token
context and up to 2,048 output tokens. The same model/configuration, rubric,
personas and conversations are used for every compared system. This controls
conditions but does not eliminate judge bias; support comparisons with manual spot-checks.
It also reports simulated would-send percentage, not real human approval.
`scripts/data-split/make_rating_kit.py` and `analyze_ratings.py` remain optional
human-spreadsheet tools. Main results and judge reliability checks are not complete.
See `split_and_rating_protocol.md` for the shared protocol.
