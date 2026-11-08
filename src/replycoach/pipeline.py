"""Shared memory, normalization and ranking for single or compared retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

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
class PreparedInput:
    message: str
    memory: list[str]
    query: str
    retrieval_memory: list[str]
    normalised_message: str
    detected_terms: list[dict]


@dataclass
class ComparisonResult:
    prepared: PreparedInput
    results: dict[str, CoachResult]
    errors: dict[str, str]


class Retriever(Protocol):
    def retrieve(self, message: str, **kwargs) -> list[Candidate]: ...


def prepare_input(message: str, session: list[str], memory_mode: str) -> PreparedInput:
    memory = select_memory(session, message, mode=memory_mode)
    detection = detect_and_normalise(message)
    return PreparedInput(
        message=message,
        memory=memory,
        query=retrieval_query(message),
        retrieval_memory=[retrieval_query(turn) for turn in memory],
        normalised_message=detection.normalised,
        detected_terms=detection.detected,
    )


def rank_result(
    prepared: PreparedInput, candidates: list[Candidate], method: str,
    *, retrieval_method: str | None = None,
) -> CoachResult:
    """Keep traceable candidate origins while applying the shared ranking policy."""
    restored = []
    for candidate in candidates:
        meta = {**candidate.meta, "candidate_source": candidate.source}
        if candidate.source in {"V3", "V4", "V5", "MT5_A", "SEALION"}:
            meta["generation_source"] = candidate.source
        else:
            meta.update(retrieval_source=candidate.source, retrieval_method=retrieval_method or method)
        restored.append(replace(
            candidate,
            context=list(prepared.memory),
            message=prepared.message,
            normalised_message=prepared.normalised_message,
            meta=meta,
        ))
    ranked = rerank(restored, prepared.message, prepared.memory)
    chosen = pick_two(ranked)
    return CoachResult(
        message=prepared.message,
        normalised_message=prepared.normalised_message,
        detected_terms=prepared.detected_terms,
        memory=prepared.memory,
        candidates=ranked,
        standard=maybe_fallback(chosen.get("STANDARD"), prepared.message, prepared.memory, "STANDARD"),
        local=maybe_fallback(chosen.get("LOCAL"), prepared.message, prepared.memory, "LOCAL"),
        method=method,
    )


def compare_replies(
    message: str,
    session: list[str],
    memory_mode: str,
    bm25: Retriever | Callable[[], Retriever],
    minilm_factory: Callable[[], Retriever],
    top_k: int = 3,
    log: bool = True,
    bank_fingerprint: str | None = None,
) -> ComparisonResult:
    """Prepare once, then run two independently recoverable retrieval paths."""
    prepared = prepare_input(message, session, memory_mode)
    results = {}
    errors = {}
    bm25_factory = bm25 if callable(bm25) else lambda: bm25
    for method, factory in (("bm25", bm25_factory), ("minilm", minilm_factory)):
        try:
            retriever = factory()
            kwargs = {"context": list(prepared.retrieval_memory), "top_k": top_k}
            if method == "bm25":
                kwargs.update(method="bm25", source="V0")
            else:
                kwargs.update(source="V2")
            candidates = retriever.retrieve(prepared.query, **kwargs)
            results[method] = rank_result(prepared, candidates, method)
        except Exception as exc:
            # A missing model must not masquerade as another retrieval method.
            errors[method] = str(exc) or type(exc).__name__
    comparison = ComparisonResult(prepared=prepared, results=results, errors=errors)
    if log:
        log_event("compare_suggestions", {
            "message": message,
            "normalised": prepared.normalised_message,
            "memory": prepared.memory,
            "bank_fingerprint": bank_fingerprint,
            "methods": {
                method: {"standard": result.standard.to_dict(), "local": result.local.to_dict()}
                for method, result in results.items()
            },
            "errors": errors,
        })
    return comparison


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
        prepared = prepare_input(message, session or [], self.memory_mode)
        retrieved = self.retriever.retrieve(
            prepared.query, context=prepared.retrieval_memory,
            top_k=self.top_k, method=self.method, source="V0",
        )
        result = rank_result(prepared, retrieved, self.method)
        if log:
            log_event(
                "suggest",
                {
                    "message": message,
                    "normalised": result.normalised_message,
                    "memory": result.memory,
                    "standard": result.standard.candidate_text,
                    "local": result.local.candidate_text,
                    "standard_source": result.standard.source,
                    "local_source": result.local.source,
                    "top_item": retrieved[0].item_id if retrieved else None,
                },
            )
        return result
