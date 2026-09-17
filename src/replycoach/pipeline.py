"""Week 4 end-to-end flow: memory -> V1 normalise -> V0 retrieve -> V6 pick two."""

from __future__ import annotations

from dataclasses import dataclass, field

from replycoach.fallback import maybe_fallback
from replycoach.io import load_items
from replycoach.logging_utils import log_event
from replycoach.memory import select_memory
from replycoach.rerank import pick_two, rerank
from replycoach.retrieval.v0 import V0Retriever
from replycoach.retrieval.v1 import detect_and_normalise, retrieval_query
from replycoach.schema import Candidate, DialogueItem


@dataclass
class CoachResult:
    message: str
    normalised_message: str
    detected_terms: list[dict]
    memory: list[str]
    candidates: list[Candidate]
    standard: Candidate
    local: Candidate
    method: str


@dataclass
class ReplyCoach:
    retriever: V0Retriever
    memory_mode: str = "M2"
    method: str = "bm25"
    top_k: int = 3
    items: list[DialogueItem] = field(default_factory=list)

    @classmethod
    def from_pilot(cls, memory_mode: str = "M2") -> "ReplyCoach":
        items = load_items()
        return cls(retriever=V0Retriever.from_items(items), memory_mode=memory_mode, items=items)

    def suggest(self, message: str, session: list[str] | None = None, log: bool = True) -> CoachResult:
        session = session or []
        memory = select_memory(session, message, mode=self.memory_mode)
        detection = detect_and_normalise(message)
        query = retrieval_query(" ".join(memory + [message]))
        retrieved = self.retriever.retrieve(query, context=memory, top_k=self.top_k, method=self.method, source="V0")
        for candidate in retrieved:
            candidate.normalised_message = detection.normalised
        ranked = rerank(retrieved, message, memory)
        chosen = pick_two(ranked)
        standard = maybe_fallback(chosen.get("STANDARD"), message, memory, "STANDARD")
        local = maybe_fallback(chosen.get("LOCAL"), message, memory, "LOCAL")
        result = CoachResult(
            message=message,
            normalised_message=detection.normalised,
            detected_terms=detection.detected,
            memory=memory,
            candidates=ranked,
            standard=standard,
            local=local,
            method=self.method,
        )
        if log:
            log_event(
                "suggest",
                {
                    "message": message,
                    "normalised": detection.normalised,
                    "memory": memory,
                    "standard": standard.candidate_text,
                    "local": local.candidate_text,
                    "standard_source": standard.source,
                    "local_source": local.source,
                    "top_item": retrieved[0].item_id if retrieved else None,
                },
            )
        return result
