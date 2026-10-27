"""Deterministic weak supervision from reviewed development pairs, never AI ratings."""
from hashlib import sha256

from replycoach.all_models import align_memory_speakers
from replycoach.evaluation import _validate_inputs
from replycoach.pipeline import prepare_input
from replycoach.schema import STYLES
from replycoach.shared_ranker import build_query
from replycoach.reranker_bundle import (BASE_MODEL, BASE_REVISION, QUERY_FORMAT, normalized,
                                       pair_payload, validate_training_bundle,
                                       write_training_bundle, load_training_bundle)


def build_training_bundle(inputs, negatives=2, seed=425):
    _validate_inputs(inputs)
    if type(negatives) is not int or not 1 <= negatives <= 8:
        raise ValueError('negative count must be between 1 and 8')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    train = sorted(inputs.bank.items, key=lambda i: i.id)
    val = sorted(inputs.queries.items, key=lambda i: i.id)
    bundle = {}
    for split, items in [('train', train), ('validation', val)]:
        rows = []
        for item in items:
            session = item.memory_texts()
            memory = prepare_input(item.message.text, session, 'M2').memory
            speakers = align_memory_speakers(session, [t.speaker for t in item.session_memory], memory)
            known = {normalized(item.replies.standard), normalized(item.replies.local)}
            for style in STYLES:
                query = build_query(item.message.text, memory, speakers, style)
                def record(donor, label):
                    return {'item_id': item.id, 'donor_id': donor.id, 'style': style,
                            'query': query, 'reply': getattr(donor.replies, style.lower()),
                            'label': label, 'provenance': 'human_reviewed_pair' if label else 'algorithmic_mismatch_weak'}
                rows.append(record(item, 1))
                donors = sorted((d for d in train if d.id != item.id), key=lambda d:
                                sha256(f'{seed}|{item.id}|{style}|{d.id}'.encode()).hexdigest())
                used, added = set(known), 0
                for donor in donors:
                    text = normalized(getattr(donor.replies, style.lower()))
                    if text in used:
                        continue
                    rows.append(record(donor, 0))
                    used.add(text)
                    added += 1
                    if added == negatives:
                        break
                if added == 0:
                    raise ValueError(f'{item.id}/{style}: no distinct training donor is available')
        bundle[split] = rows
    bundle['manifest'] = {
        'schema_version': 1, 'base_model': BASE_MODEL, 'base_revision': BASE_REVISION,
        'query_format': QUERY_FORMAT, 'memory_mode': 'M2', 'seed': seed,
        'negative_count_requested': negatives,
        'negative_policy': 'algorithmic mismatches are weak labels; false negatives may remain; no LLM labels',
        'train_ids': [i.id for i in train], 'validation_ids': [i.id for i in val],
        'train_fingerprint': inputs.bank.fingerprint, 'validation_fingerprint': inputs.queries.fingerprint,
        'pair_sha256': {s: sha256(pair_payload(bundle[s])).hexdigest() for s in ['train', 'validation']},
    }
    return validate_training_bundle(bundle)
