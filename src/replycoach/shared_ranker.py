"""Frozen, source-neutral scoring of the existing per-model reply proposals."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Protocol

from replycoach.schema import Candidate, STYLES
from replycoach.reranker_bundle import QUERY_FORMAT as STYLE_QUERY_FORMAT

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


def build_query(message: str, memory: list[str], speakers: list[str] | None = None,
                style: str | None = None) -> str:
    if not message.strip():
        raise ValueError('incoming message must be nonempty')
    if speakers is not None and len(speakers) != len(memory):
        raise ValueError('speaker count must match memory turns')
    names = {'peer': 'Peer', 'user': 'You', 'other': 'Other', 'unknown': 'Unknown'}
    roles = speakers if speakers is not None else ['unknown'] * len(memory)
    if any(role not in names for role in roles):
        raise ValueError('unsupported memory speaker')
    if style is not None and style not in STYLES:
        raise ValueError('unsupported reply style')
    turns = [f'{names[role]}: {text}' for role, text in zip(roles, memory)]
    return '\n'.join(['Conversation (oldest first):', *turns,
                      f'Incoming peer message: {message}',
                      *([f'Requested reply style: {style}'] if style else []), 'User reply:'])


def read_finetuned_manifest(checkpoint):
    path = Path(checkpoint)
    manifest = json.loads((path/'v6_training.json').read_text(encoding='utf-8'))
    if (manifest.get('record_type') != 'v6_cross_encoder_training'
            or manifest.get('completed') is not True or manifest.get('production_checkpoint') is not True
            or manifest.get('device') != 'cuda' or manifest.get('weights_changed') is not True
            or type(manifest.get('optimizer_steps')) is not int or manifest['optimizer_steps'] < 1
            or manifest.get('query_format') != STYLE_QUERY_FORMAT
            or manifest.get('base_model') != MODEL_NAME or manifest.get('base_revision') != MODEL_REVISION
            or manifest.get('initial_weights_sha256') == manifest.get('best_weights_sha256')
            or not isinstance(manifest.get('best_weights_sha256'), str)
            or len(manifest['best_weights_sha256']) != 64):
        raise ValueError('a completed GPU checkpoint with valid V6 provenance is required; smoke models are excluded')
    return manifest


class CrossEncoderScorer:
    """One small pinned CPU model; downloaded weights only, never hosted inference."""

    def __init__(self, checkpoint=None, *, style_aware=False):
        manifest = read_finetuned_manifest(checkpoint) if checkpoint is not None else None
        if manifest:
            for name in ['model.safetensors', 'config.json', 'tokenizer.json']:
                if not (Path(checkpoint)/name).is_file():
                    raise FileNotFoundError(f'V6 checkpoint file missing: {name}')
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.identity = {'model': MODEL_NAME, 'revision': MODEL_REVISION,
                         'query_format': QUERY_FORMAT, 'max_length': 512, 'device': 'cpu'}
        source = Path(checkpoint) if manifest else MODEL_NAME
        kwargs = {'local_files_only': True} if manifest else {'revision': MODEL_REVISION}
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
        self.tokenizer.truncation_side = 'left'
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source, trust_remote_code=False, use_safetensors=True, **kwargs).to('cpu').eval()
        if manifest:
            from replycoach.reranker_fit import weights_sha256
            if weights_sha256(self.model) != manifest['best_weights_sha256']:
                raise ValueError('V6 weights do not match the training provenance hash')
            self.identity.update(model='replycoach-v6-finetuned-MiniLM', revision=manifest['best_weights_sha256'],
                                 base_model=MODEL_NAME, base_revision=MODEL_REVISION, fine_tuned=True,
                                 best_epoch=manifest['best_epoch'],
                                 train_fingerprint=manifest['train_fingerprint'],
                                 validation_fingerprint=manifest['validation_fingerprint'])
        if manifest or style_aware:
            self.identity['query_format'] = STYLE_QUERY_FORMAT

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
    style_aware = scorer.identity.get('query_format') == STYLE_QUERY_FORMAT
    scores = list(scorer.score_pairs([
        (build_query(message, memory, speakers, style) if style_aware else query, text)
        for style, text in keys]))
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
        meta = {'selector': 'finetuned_cross_encoder' if scorer.identity.get('fine_tuned') else 'shared_cross_encoder',
                'ranker': dict(scorer.identity),
                'shared_rank_score': float(value.score), 'score_kind': 'cross_encoder_logit',
                'truncated': bool(value.truncated), 'contributing_models': models,
                'fallback_models': sorted({o['model'] for o in originals if o['origin'] == 'fallback'}),
                'original_proposals': originals}
        ranked.append(replace(chosen, context=list(memory), message=message, source='V6',
                              score=float(value.score), meta=meta))
    ranked.sort(key=lambda c: (STYLES.index(c.style), -c.score, ' '.join(c.candidate_text.split())))
    selected = {style: next((c for c in ranked if c.style == style), None) for style in STYLES}
    return SharedRanking(ranked, selected['STANDARD'], selected['LOCAL'])
