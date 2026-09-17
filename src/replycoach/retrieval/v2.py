"""Week 6 stub: static embedding retrieval. Do not import this in Week 4 demos."""

from __future__ import annotations

# Planned recipe:
# 1. Encode message + memory with a local MiniLM (sentence-transformers), no API.
# 2. Encode each pilot/dev item the same way and cache vectors in artifacts/.
# 3. Rank by cosine similarity, then emit STANDARD/LOCAL candidates in the shared format.
# 4. Compare against V0 BM25 on development data only.

PLACEHOLDER = "V2 embedding retriever lands in Week 6."
