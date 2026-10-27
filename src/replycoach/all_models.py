"""Compare eight existing source proposals with one source-neutral scorer."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from replycoach.logging_utils import log_event
from replycoach.pipeline import PreparedInput, prepare_input, rank_result
from replycoach.schema import Candidate
from replycoach.shared_ranker import PairScorer, SharedRanking, build_query, rank_proposals
from replycoach.systems import _generate, _retrieve

SOURCE_NAMES = ('BM25', 'MiniLM', 'V4', 'V5')


def align_memory_speakers(session: list[str], speakers: list[str] | None,
                          memory: list[str]) -> list[str]:
    """Align chronological selected memory backwards, retaining repeated turns."""
    if speakers is not None and len(speakers) != len(session):
        raise ValueError('speaker count must match conversation turns')
    roles = speakers if speakers is not None else ['unknown'] * len(session)
    result, position = [], len(session) - 1
    for text in reversed(memory):
        while position >= 0 and session[position] != text:
            position -= 1
        if position < 0:
            raise ValueError('selected memory does not align with conversation speakers')
        result.append(roles[position])
        position -= 1
    return list(reversed(result))


@dataclass
class AllModelsResult:
    prepared: PreparedInput
    proposals: dict[str, list[Candidate]]
    ranking: SharedRanking | None
    source_errors: dict[str, str]
    ranking_error: str | None
    source_seconds: dict[str, float]
    ranking_seconds: float
    memory_speakers: list[str]

    @property
    def available_sources(self) -> list[str]:
        return [name for name in SOURCE_NAMES if name in self.proposals]

    @property
    def complete(self) -> bool:
        return (len(self.proposals) == 4 and not self.source_errors
                and self.ranking is not None and self.ranking.standard is not None
                and self.ranking.local is not None and self.ranking_error is None)

    def to_dict(self) -> dict:
        return {'record_type': 'all_models_comparison', 'complete': self.complete,
                'requested_sources': list(SOURCE_NAMES), 'available_sources': self.available_sources,
                'message': self.prepared.message, 'memory': self.prepared.memory,
                'memory_speakers': self.memory_speakers, 'source_errors': dict(self.source_errors),
                'ranking_error': self.ranking_error, 'source_seconds': dict(self.source_seconds),
                'ranking_seconds': self.ranking_seconds,
                'proposals': {name: [c.to_dict() for c in candidates] for name, candidates in self.proposals.items()},
                'ranked_candidates': [c.to_dict() for c in self.ranking.candidates] if self.ranking else [],
                'standard': self.ranking.standard.to_dict() if self.ranking and self.ranking.standard else None,
                'local': self.ranking.local.to_dict() if self.ranking and self.ranking.local else None}


def run_all_models(message: str, session: list[str], memory_mode: str = 'M2', *,
                   bm25_factory: Callable | None = None, minilm_factory: Callable | None = None,
                   v4_factory: Callable | None = None, v5_factory: Callable | None = None,
                   scorer_factory: Callable[[], PairScorer] | None = None,
                   speakers: list[str] | None = None, log: bool = False) -> AllModelsResult:
    build_query(message, session, speakers)  # Validate before loading expensive resources.
    prepared = prepare_input(message, session, memory_mode)
    memory_speakers = align_memory_speakers(session, speakers, prepared.memory)
    proposals, errors, times = {}, {}, {}
    for name, factory, method in [('BM25', bm25_factory, 'bm25'), ('MiniLM', minilm_factory, 'minilm'),
                                   ('V4', v4_factory, None), ('V5', v5_factory, None)]:
        started = perf_counter()
        try:
            candidates = (_retrieve(prepared, factory, method, 3) if method
                          else _generate(prepared, factory))
            result = rank_result(prepared, candidates, 'retrieval_only' if method else 'generator_only',
                                 retrieval_method=method)
            proposals[name] = [result.standard, result.local]
        except Exception as exc:
            errors[name] = str(exc) or type(exc).__name__
        times[name] = perf_counter() - started
    ranking, ranking_error, ranking_seconds = None, None, 0.
    if proposals:
        started = perf_counter()
        try:
            if scorer_factory is None:
                raise ValueError('shared ranker is not configured')
            ranking = rank_proposals(message, prepared.memory, proposals, scorer_factory(), memory_speakers)
        except Exception as exc:
            ranking_error = str(exc) or type(exc).__name__
        ranking_seconds = perf_counter() - started
    output = AllModelsResult(prepared, proposals, ranking, errors, ranking_error,
                             times, ranking_seconds, memory_speakers)
    if log:
        log_event('all_models_suggestions', output.to_dict())
    return output
