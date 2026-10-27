"""Frozen, source-neutral scoring of the existing per-model reply proposals."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Protocol

from replycoach.schema import Candidate, STYLES

MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L6-v2'
MODEL_REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'
QUERY_FORMAT = 'speaker-conversation-v1'


@dataclass(frozen=True)
class PairScore:
    score: float
    truncated: bool = False


class PairScorer(Protocol):
    identity: dict

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[PairScore]: ...


@dataclass
class SharedRanking:
    candidates: list[Candidate]
    standard: Candidate | None
    local: Candidate | None


def build_query(message: str, memory: list[str], speakers: list[str] | None = None) -> str:
    if not message.strip():
        raise ValueError('incoming message must be nonempty')
    if speakers is not None and len(speakers) != len(memory):
        raise ValueError('speaker count must match memory turns')
    names = {'peer': 'Peer', 'user': 'You', 'other': 'Other', 'unknown': 'Unknown'}
    roles = speakers if speakers is not None else ['unknown'] * len(memory)
    if any(role not in names for role in roles):
        raise ValueError('unsupported memory speaker')
    turns = [f'{names[role]}: {text}' for role, text in zip(roles, memory)]
    return '\n'.join(['Conversation (oldest first):', *turns,
                      f'Incoming peer message: {message}', 'User reply:'])


class CrossEncoderScorer:
    """One small pinned CPU model; downloaded weights only, never hosted inference."""

    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.identity = {'model': MODEL_NAME, 'revision': MODEL_REVISION,
                         'query_format': QUERY_FORMAT, 'max_length': 512, 'device': 'cpu'}
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION,
                                                       trust_remote_code=False)
        self.tokenizer.truncation_side = 'left'
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=False).to('cpu').eval()

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[PairScore]:
        import torch

        result = []
        for start in range(0, len(pairs), 8):
            batch = pairs[start:start + 8]
            queries, replies = map(list, zip(*batch))
            lengths = self.tokenizer(queries, replies, truncation=False, return_length=True)['length']
            encoded = self.tokenizer(queries, replies, padding=True, truncation='only_first',
                                     max_length=512, return_tensors='pt')
            with torch.inference_mode():
                values = self.model(**encoded).logits
            if values.ndim != 2 or values.shape[1] != 1:
                raise ValueError('shared ranker must return one score per pair')
            result.extend(PairScore(float(score), length > 512)
                          for score, length in zip(values[:, 0].tolist(), lengths))
        return result


def rank_proposals(message: str, memory: list[str], proposals: dict[str, list[Candidate]],
                   scorer: PairScorer, speakers: list[str] | None = None) -> SharedRanking:
    """Keep styles separate; original source scores never influence shared selection."""
    query = build_query(message, memory, speakers)
    grouped: dict[tuple[str, str], list[tuple[str, Candidate]]] = {}
    for model, candidates in proposals.items():
        for candidate in candidates:
            candidate.validate()
            key = (candidate.style, ' '.join(candidate.candidate_text.split()))
            grouped.setdefault(key, []).append((model, candidate))
    keys = sorted(grouped, key=lambda key: (STYLES.index(key[0]), key[1]))
    if not keys:
        return SharedRanking([], None, None)
    scores = list(scorer.score_pairs([(query, text) for _, text in keys]))
    if len(scores) != len(keys):
        raise ValueError('shared ranker score count does not match candidate count')
    ranked = []
    for (style, text), value in zip(keys, scores):
        if (not isinstance(value, PairScore) or isinstance(value.score, bool)
                or not isinstance(value.score, (int, float)) or not math.isfinite(value.score)):
            raise ValueError('shared ranker scores must be finite numbers')
        entries = grouped[(style, text)]
        originals = []
        for model, candidate in sorted(entries, key=lambda entry: (entry[0], entry[1].candidate_text)):
            originals.append({'model': model, 'origin': candidate.meta.get('candidate_source', candidate.source),
                              'score': float(candidate.score), 'item_id': candidate.item_id,
                              'reply': candidate.candidate_text})
        models = sorted({model for model, _ in entries})
        chosen = min((c for _, c in entries), key=lambda c: c.candidate_text)
        meta = {'selector': 'shared_cross_encoder', 'ranker': dict(scorer.identity),
                'shared_rank_score': float(value.score), 'score_kind': 'cross_encoder_logit',
                'truncated': bool(value.truncated), 'contributing_models': models,
                'fallback_models': sorted({o['model'] for o in originals if o['origin'] == 'fallback'}),
                'original_proposals': originals}
        ranked.append(replace(chosen, context=list(memory), message=message, source='V6',
                              score=float(value.score), meta=meta))
    ranked.sort(key=lambda c: (STYLES.index(c.style), -c.score, ' '.join(c.candidate_text.split())))
    selected = {style: next((c for c in ranked if c.style == style), None) for style in STYLES}
    return SharedRanking(ranked, selected['STANDARD'], selected['LOCAL'])
