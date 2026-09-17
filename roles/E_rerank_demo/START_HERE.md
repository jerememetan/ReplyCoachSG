# Role E — Reranker, integration, demo

## This week (Week 4)

Shared candidate format lives in `src/replycoach/schema.py` (`Candidate`).

Demo:

```powershell
python -m replycoach.demo_cli
streamlit run src/replycoach/demo_app.py
```

Logging: `src/replycoach/logging_utils.py` → `experiments/logs/runs.jsonl`.

A first V6 baseline already rescores with token overlap (`src/replycoach/rerank.py`). Improve it in Week 6; do not replace the format.

## Next

- Week 5: nicer candidate table in the demo.
- Week 6: use retrieval score + message similarity.
- Week 7: include generator candidates.
- Week 8: full flow with fallback and two replies.
- Week 11: presentation demo on 20 unseen conversations.

## Friday artefact

A screenshot or CLI printout showing STANDARD and LOCAL from V0.
