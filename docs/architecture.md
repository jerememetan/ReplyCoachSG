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
