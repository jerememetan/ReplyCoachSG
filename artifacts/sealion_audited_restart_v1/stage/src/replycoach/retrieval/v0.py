"""V0 retrieval: BM25 primary, TF-IDF cosine as a comparable baseline."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from replycoach.retrieval import BM25Index, build_index
from replycoach.schema import Candidate, DialogueItem
from replycoach.tokenize import tokenize_join


def _index_text(item: DialogueItem) -> str:
    memory = " ".join(item.memory_texts())
    return " ".join([memory, item.message.text, item.replies.standard, item.replies.local])


@dataclass
class V0Retriever:
    items: list[DialogueItem]
    bm25: BM25Index
    vectorizer: TfidfVectorizer
    tfidf_matrix: object

    @classmethod
    def from_items(cls, items: list[DialogueItem]) -> "V0Retriever":
        corpus = [_index_text(item) for item in items]
        bm25 = build_index(corpus)
        vectorizer = TfidfVectorizer(preprocessor=tokenize_join, tokenizer=str.split, token_pattern=None)
        matrix = vectorizer.fit_transform(corpus)
        return cls(items=items, bm25=bm25, vectorizer=vectorizer, tfidf_matrix=matrix)

    def _tfidf_rank(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        ranked = sorted(enumerate(sims), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]

    def retrieve(
        self,
        message: str,
        context: list[str] | None = None,
        top_k: int = 3,
        method: str = "bm25",
        source: str = "V0",
    ) -> list[Candidate]:
        query = " ".join((context or []) + [message])
        if method == "bm25":
            ranked = self.bm25.rank(query, top_k=max(top_k, len(self.items)))
            peak = max((score for _, score in ranked), default=1.0) or 1.0
            ranked = [(idx, score / peak) for idx, score in ranked[:top_k]]
        else:
            ranked = self._tfidf_rank(query, top_k)
        candidates: list[Candidate] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            item = self.items[idx]
            meta = {"rank": rank, "method": method, "scenario": item.scenario, "intent": item.intent}
            candidates.append(
                Candidate(
                    context=list(context or item.memory_texts()),
                    message=message,
                    candidate_text=item.replies.standard,
                    style="STANDARD",
                    source=source,
                    score=float(score),
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
                    score=float(score),
                    item_id=item.id,
                    meta=meta,
                )
            )
        for candidate in candidates:
            candidate.validate()
        return candidates
