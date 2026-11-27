"""Model-independent retrieval, generation and hybrid execution."""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from typing import Callable

from replycoach.generate.providers import GeneratorUnavailable, ReplyGenerator
from replycoach.logging_utils import log_event
from replycoach.pipeline import CoachResult, PreparedInput, Retriever, prepare_input, rank_result
from replycoach.schema import Candidate, STYLES

SYSTEM_MODES = ("retrieval_only", "generator_only", "hybrid")
DECODER_TOKEN = re.compile(r"<(?:unk|pad|bos|eos|s|/s|extra_id_\d+|standard|local)>", re.IGNORECASE)


@dataclass
class SystemResult:
    requested_mode: str
    effective_mode: str
    prepared: PreparedInput
    result: CoachResult
    errors: dict[str, str]
    retrieval_method: str
    retrieval_count: int
    generation_count: int

    def to_dict(self) -> dict:
        return {
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "message": self.prepared.message,
            "normalised_message": self.prepared.normalised_message,
            "memory": self.prepared.memory,
            "retrieval_method": self.retrieval_method,
            "retrieval_count": self.retrieval_count,
            "generation_count": self.generation_count,
            "errors": dict(self.errors),
            "standard": self.result.standard.to_dict(),
            "local": self.result.local.to_dict(),
        }


@dataclass
class SystemComparison:
    prepared: PreparedInput
    results: dict[str, SystemResult]


def _retrieve(prepared: PreparedInput, factory: Callable[[], Retriever] | None, method: str, top_k: int):
    if factory is None:
        raise ValueError("retriever is not configured")
    kwargs = {"context": list(prepared.retrieval_memory), "top_k": top_k}
    if method == "bm25":
        kwargs.update(method="bm25", source="V0")
    else:
        kwargs.update(source="V2")
    candidates = list(factory().retrieve(prepared.query, **kwargs))
    if not candidates:
        raise ValueError("retriever returned no candidates")
    for candidate in candidates:
        candidate.validate()
        if not math.isfinite(float(candidate.score)):
            raise ValueError("retrieval score must be finite")
    return candidates


def _generate(prepared: PreparedInput, factory: Callable[[], ReplyGenerator] | None):
    if factory is None:
        raise GeneratorUnavailable("No trained generator is configured; see docs/role_handoffs.md.")
    generator = factory()
    if generator.source not in {"V4", "V5"}:
        raise ValueError("generator source must be V4 or V5")
    candidates = generator.generate(prepared.message, list(prepared.memory))
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("generator must return nonempty Candidate outputs")
    for candidate in candidates:
        if not isinstance(candidate, Candidate):
            raise ValueError("generator outputs must be Candidate objects")
        candidate.validate()
        if not isinstance(candidate.candidate_text, str) or not candidate.candidate_text.strip():
            raise ValueError("generated text must be a nonempty string")
        if DECODER_TOKEN.search(candidate.candidate_text):
            raise ValueError("generated text contains unresolved decoder tokens")
        if candidate.source != generator.source or candidate.item_id is not None:
            raise ValueError("generated candidates must use the model source and no stored item_id")
        if isinstance(candidate.score, bool) or not isinstance(candidate.score, (int, float)):
            raise ValueError("generator score must be a number in [0, 1]")
        if not math.isfinite(candidate.score) or not 0 <= candidate.score <= 1:
            raise ValueError("generator score must be finite and in [0, 1]")
        if not isinstance(candidate.meta, dict):
            raise ValueError("generated candidate metadata must be a dictionary")
        json.dumps(candidate.to_dict(), allow_nan=False)
    if {candidate.style for candidate in candidates} != set(STYLES):
        raise ValueError("generator must return both STANDARD and LOCAL styles")
    return list(candidates)


def _run_modes(
    message: str, session: list[str], memory_mode: str, modes: tuple[str, ...],
    retrieval_method: str, retriever_factory: Callable[[], Retriever] | None,
    generator_factory: Callable[[], ReplyGenerator] | None, top_k: int,
) -> SystemComparison:
    if not modes or any(mode not in SYSTEM_MODES for mode in modes):
        raise ValueError(f"system mode must be one of {SYSTEM_MODES}")
    if retrieval_method not in {"bm25", "minilm"}:
        raise ValueError("retrieval method must be bm25 or minilm")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not message.strip():
        raise ValueError("incoming message must be nonempty")
    prepared = prepare_input(message, session, memory_mode)
    retrieved, generated, errors = [], [], {}
    if any(mode in {"retrieval_only", "hybrid"} for mode in modes):
        try:
            retrieved = _retrieve(prepared, retriever_factory, retrieval_method, top_k)
        except Exception as exc:
            errors["retrieval"] = str(exc) or type(exc).__name__
    if any(mode in {"generator_only", "hybrid"} for mode in modes):
        try:
            generated = _generate(prepared, generator_factory)
        except Exception as exc:
            errors["generator"] = str(exc) or type(exc).__name__
    results = {}
    for mode in modes:
        r = retrieved if mode in {"retrieval_only", "hybrid"} else []
        g = generated if mode in {"generator_only", "hybrid"} else []
        effective = "hybrid" if r and g else "retrieval_only" if r else "generator_only" if g else "fallback_only"
        mode_errors = {source: error for source, error in errors.items()
                       if (source == "retrieval" and mode != "generator_only")
                       or (source == "generator" and mode != "retrieval_only")}
        result = rank_result(prepared, r + g, effective, retrieval_method=retrieval_method)
        results[mode] = SystemResult(mode, effective, prepared, result, mode_errors,
                                     retrieval_method, len(r), len(g))
    return SystemComparison(prepared, results)


def run_system(
    message: str, session: list[str], memory_mode: str = "M2", *, mode: str = "hybrid",
    retrieval_method: str = "bm25", retriever_factory: Callable[[], Retriever] | None = None,
    generator_factory: Callable[[], ReplyGenerator] | None = None, top_k: int = 3,
    log: bool = False,
) -> SystemResult:
    output = _run_modes(message, session, memory_mode, (mode,), retrieval_method,
                        retriever_factory, generator_factory, top_k).results[mode]
    if log:
        log_event("system_suggestions", output.to_dict())
    return output


def compare_system_modes(
    message: str, session: list[str], memory_mode: str = "M2", *,
    retrieval_method: str = "bm25", retriever_factory: Callable[[], Retriever] | None = None,
    generator_factory: Callable[[], ReplyGenerator] | None = None, top_k: int = 3,
) -> SystemComparison:
    """Compare pools fairly: one prepared input and one call to each source."""
    return _run_modes(message, session, memory_mode, SYSTEM_MODES, retrieval_method,
                      retriever_factory, generator_factory, top_k)
