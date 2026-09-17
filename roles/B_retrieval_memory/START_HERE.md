# Role B — Retrieval and memory

## This week (Week 4)

V0 is implemented:

```powershell
python -m replycoach.demo_cli "eh later go makan? can 吃饭 then library"
```

Code: `src/replycoach/retrieval/v0.py`, `src/replycoach/retrieval/__init__.py` (BM25), `src/replycoach/memory.py`.

## Next

- Week 5: own V1 lexicon edits in `data/lexicons/local_terms.json`. Ambiguous words must stay `"apply": false`.
- Week 6: implement V2 MiniLM embeddings in `src/replycoach/retrieval/v2.py`. Keep it local, no APIs.
- Week 8: measure M0 vs M1 vs M2 on development data.

## Friday artefact

Top-3 retrieved pairs for 5 held-in pilot queries, saved under `experiments/logs/`.
