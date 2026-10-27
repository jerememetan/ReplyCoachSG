"""Training-only hard-example mining and explicit, stale-safe human review."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile

from filelock import FileLock

from replycoach.evaluation import validate_output_path
from replycoach.reranker_bundle import normalized, validate_training_bundle
from replycoach.reranker_hard_pairs import apply_reviewed_pairs

DECISIONS = ('pending', 'suitable', 'unsuitable', 'uncertain', 'reference_needs_correction')
MUTABLE = {'decision', 'reference_ok', 'reviewer', 'notes', 'reviewed_at'}
CASE_KEYS = {'id', 'item_id', 'donor_id', 'style', 'query', 'reference_reply', 'alternative_reply',
             'reference_score', 'alternative_score'} | MUTABLE


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')


def _identity(batch):
    return sha256(_encoded({'metadata': batch['metadata'], 'comparisons': [
        {k: v for k, v in r.items() if k not in MUTABLE} for r in batch['comparisons']]})).hexdigest()


def _pair(row, label):
    return {'item_id': row['item_id'], 'donor_id': row['donor_id'], 'style': row['style'],
            'query': row['query'], 'reply': row['alternative_reply'], 'label': label,
            'provenance': 'human_reviewed_alternative'}


def _supplement(batch, pairs, skipped):
    meta = batch['metadata']
    return {'schema_version': 1, 'record_type': 'v6_reviewed_hard_pairs',
            **{k: meta[k] for k in ['train_fingerprint', 'validation_fingerprint', 'base_pair_sha256']},
            'source_review_sha256': sha256(_encoded(batch)).hexdigest(),
            'comparison_count': len(batch['comparisons']), 'skipped_count': skipped, 'pairs': pairs}


def validate_review_batch(batch, bundle):
    validate_training_bundle(bundle)
    if (set(batch) != {'record_type', 'metadata', 'comparisons', 'identity_sha256'}
            or batch['record_type'] != 'v6_hard_review_batch' or not batch['comparisons']
            or batch['identity_sha256'] != _identity(batch)):
        raise ValueError('invalid or changed comparison batch identity')
    positives = {(p['item_id'], p['style']): p for p in bundle['train'] if p['label']}
    seen = set()
    for row in batch['comparisons']:
        source = positives.get((row['item_id'], row['style']))
        if (set(row) != CASE_KEYS or row['id'] in seen or source is None
                or row['reference_reply'] != source['reply'] or row['decision'] not in DECISIONS
                or type(row['reference_ok']) is not bool
                or any(not isinstance(row[k], str) for k in ['reviewer', 'notes', 'reviewed_at'])
                or any(type(row[k]) not in {float, int} or not math.isfinite(row[k])
                       for k in ['reference_score', 'alternative_score'])):
            raise ValueError('invalid comparison fields or reference')
        seen.add(row['id'])
        if row['decision'] != 'pending' and (not row['reviewer'].strip() or not row['reviewed_at']):
            raise ValueError('completed review needs a named reviewer and timestamp')
        if row['decision'] in {'suitable', 'unsuitable'} and not row['reference_ok']:
            raise ValueError('confirm the reference reply before assigning a training label')
    # Validate immutable train-only query/donor provenance even while labels are pending.
    apply_reviewed_pairs(bundle, _supplement(batch, [_pair(r, 0) for r in batch['comparisons']], 0))
    return batch


def mine_review_batch(bundle, retriever, scorer, *, count=20, shortlist=4):
    validate_training_bundle(bundle)
    if type(count) is not int or not 2 <= count <= 100 or count % 2:
        raise ValueError('review count must be even and between 2 and 100')
    if type(shortlist) is not int or not 1 <= shortlist <= 10:
        raise ValueError('shortlist must be between 1 and 10')
    positives = {(p['item_id'], p['style']): p for p in bundle['train'] if p['label']}
    pairs, positions = [], []
    for key in sorted(positives):
        source = positives[key]
        known = {normalized(positives[key[0], style]['reply']) for style in ['STANDARD', 'LOCAL']}
        donors, seen = [], set(known)
        candidates = retriever.retrieve(source['query'], top_k=shortlist+1)
        for c in sorted(candidates, key=lambda c: (-c.score, c.item_id, c.style, c.candidate_text)):
            donor = positives.get((c.item_id, c.style))
            if donor is None or c.candidate_text != donor['reply']:
                raise ValueError('mining must use only unchanged training reply donors')
            text = normalized(c.candidate_text)
            if c.style != source['style'] or c.item_id == source['item_id'] or text in seen:
                continue
            seen.add(text)
            donors.append(donor)
            if len(donors) == shortlist:
                break
        if not donors:
            continue
        start = len(pairs)
        pairs.extend((source['query'], p['reply']) for p in [source, *donors])
        positions.append((source, donors, start))
    scores = list(scorer.score_pairs(pairs))
    if len(scores) != len(pairs) or any(not math.isfinite(float(s.score)) for s in scores):
        raise ValueError('mining scorer returned invalid scores')
    pool = []
    for source, donors, start in positions:
        best = min(range(len(donors)), key=lambda i: (-scores[start+i+1].score, donors[i]['item_id']))
        donor = donors[best]
        identity = f"{source['item_id']}|{source['style']}|{donor['item_id']}"
        pool.append({'id': 'HARD-'+sha256(identity.encode()).hexdigest()[:12],
            'item_id': source['item_id'], 'donor_id': donor['item_id'], 'style': source['style'],
            'query': source['query'], 'reference_reply': source['reply'], 'alternative_reply': donor['reply'],
            'reference_score': float(scores[start].score), 'alternative_score': float(scores[start+best+1].score),
            'decision': 'pending', 'reference_ok': False, 'reviewer': '', 'notes': '', 'reviewed_at': ''})
    pool.sort(key=lambda r: (r['reference_score']-r['alternative_score'], r['item_id'], r['style']))
    selected, used = [], set()
    for _ in range(count//2):
        for style in ['STANDARD', 'LOCAL']:
            row = next((r for r in pool if r['style'] == style and r['item_id'] not in used), None)
            if row is None:
                raise ValueError('not enough distinct training conversations for this review count')
            selected.append(row)
            used.add(row['item_id'])
    meta = bundle['manifest']
    batch = {'record_type': 'v6_hard_review_batch', 'metadata': {
        'train_fingerprint': meta['train_fingerprint'], 'validation_fingerprint': meta['validation_fingerprint'],
        'base_pair_sha256': meta['pair_sha256'], 'miner': dict(scorer.identity)}, 'comparisons': selected}
    batch['identity_sha256'] = _identity(batch)
    return validate_review_batch(batch, bundle)


def create_review_file(path, batch, bundle):
    validate_review_batch(batch, bundle)
    path = validate_output_path(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(_encoded(batch).decode('utf-8'))


def load_review_file(path, bundle):
    return validate_review_batch(json.loads(Path(path).read_text(encoding='utf-8')), bundle)


def save_decision(path, bundle, expected, reviewer, decision, reference_ok, notes):
    path = Path(path)
    reviewer = reviewer.strip()
    if not reviewer or decision not in DECISIONS[1:]:
        raise ValueError('enter your name and choose a review decision')
    if decision in {'suitable', 'unsuitable'} and reference_ok is not True:
        raise ValueError('confirm that the reference reply fits this conversation')
    with FileLock(str(path)+'.lock'):
        batch = load_review_file(path, bundle)
        row = next((r for r in batch['comparisons'] if r['id'] == expected['id']), None)
        if row != expected:
            raise ValueError('comparison changed since opening; reload before saving')
        if row['reviewer'] and row['reviewer'] != reviewer:
            raise ValueError('comparison already belongs to another reviewer')
        row.update(decision=decision, reference_ok=reference_ok, reviewer=reviewer,
                   notes=notes.strip(), reviewed_at=datetime.now(timezone.utc).isoformat())
        validate_review_batch(batch, bundle)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                              prefix='.review-', suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(_encoded(batch).decode('utf-8'))
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return batch


def export_reviewed_pairs(batch, bundle):
    validate_review_batch(batch, bundle)
    if any(r['decision'] == 'pending' for r in batch['comparisons']):
        raise ValueError('review every comparison before exporting; uncertain cases can be skipped')
    pairs = [_pair(r, int(r['decision'] == 'suitable')) for r in batch['comparisons']
             if r['decision'] in {'suitable', 'unsuitable'}]
    payload = _supplement(batch, pairs, len(batch['comparisons'])-len(pairs))
    apply_reviewed_pairs(bundle, payload)
    return payload
