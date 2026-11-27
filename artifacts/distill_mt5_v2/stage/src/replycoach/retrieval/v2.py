"""V2: local MiniLM embedding retrieval with reusable item embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from replycoach.schema import Candidate, DialogueItem, require_items

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = Path("artifacts")
CACHE_FILE = CACHE_DIR / "v2_embeddings.npy"


def _item_text(item: DialogueItem) -> str:
    """Build the text that represents one stored dialogue example."""
    memory = " ".join(item.memory_texts())
    return " ".join(
        [
            memory,
            item.message.text,
            item.replies.standard,
            item.replies.local,
        ]
    )


@dataclass
class V2Retriever:
    items: list[DialogueItem]
    model: SentenceTransformer
    embeddings: np.ndarray

    @classmethod
    def from_items(
        cls,
        items: list[DialogueItem],
        cache_file: Path | None = CACHE_FILE,
        model: SentenceTransformer | None = None,
    ) -> "V2Retriever":
        """Create a V2 retriever and cache embeddings for the supplied items."""
        items = require_items(items)

        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL_NAME)

        texts = [_item_text(item) for item in items]

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, embeddings)

        return cls(
            items=items,
            model=model,
            embeddings=embeddings,
        )

    def retrieve(
        self,
        message: str,
        context: list[str] | None = None,
        top_k: int = 3,
        source: str = "V2",
    ) -> list[Candidate]:
        """Retrieve the most semantically similar stored dialogue examples."""
        if top_k <= 0:
            return []

        query = " ".join((context or []) + [message]).strip()

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_embedding

        ranked_indices = np.argsort(-scores)[:top_k]

        candidates: list[Candidate] = []

        for rank, idx in enumerate(ranked_indices, start=1):
            item = self.items[int(idx)]
            score = float(scores[int(idx)])

            meta = {
                "rank": rank,
                "method": "embedding",
                "model": MODEL_NAME,
                "scenario": item.scenario,
                "intent": item.intent,
            }

            candidates.append(
                Candidate(
                    context=list(context or item.memory_texts()),
                    message=message,
                    candidate_text=item.replies.standard,
                    style="STANDARD",
                    source=source,
                    score=score,
                    item_id=item.id,
                    meta=meta,
                )
            )

            candidates.append(
                Candidate(
                    context=list(context or item.memory_texts()),
                    message=message,
                    candidate_text=item.replies.local,
                    style="LOCAL",
                    source=source,
                    score=score,
                    item_id=item.id,
                    meta=meta,
                )
            )

        for candidate in candidates:
            candidate.validate()

        return candidates
