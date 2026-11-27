"""Okapi BM25 over mixed English–Mandarin tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass

from replycoach.tokenize import tokenize


@dataclass
class BM25Index:
    documents: list[list[str]]
    k1: float = 1.5
    b: float = 0.75
    avgdl: float = 0.0
    df: dict[str, int] | None = None
    idf: dict[str, float] | None = None

    def __post_init__(self) -> None:
        n = len(self.documents)
        self.avgdl = sum(len(doc) for doc in self.documents) / max(n, 1)
        df: dict[str, int] = {}
        for doc in self.documents:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1
        self.df = df
        self.idf = {
            token: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        doc = self.documents[doc_index]
        if not doc:
            return 0.0
        tf: dict[str, int] = {}
        for token in doc:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        length_norm = self.k1 * (1 - self.b + self.b * len(doc) / max(self.avgdl, 1e-9))
        for token in query_tokens:
            if token not in tf:
                continue
            idf = (self.idf or {}).get(token, 0.0)
            freq = tf[token]
            score += idf * (freq * (self.k1 + 1.0)) / (freq + length_norm)
        return score

    def rank(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        scored = [(idx, self.score(query_tokens, idx)) for idx in range(len(self.documents))]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def build_index(texts: list[str], k1: float = 1.5, b: float = 0.75) -> BM25Index:
    return BM25Index(documents=[tokenize(text) for text in texts], k1=k1, b=b)
