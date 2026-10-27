import copy
import importlib
import importlib.util
from hashlib import sha256

import pytest

from test_reranker_training import inputs
from replycoach.reranker_training import build_training_bundle
from replycoach.reranker_bundle import PAIR_KEYS, pair_payload


def api():
    assert importlib.util.find_spec('replycoach.reranker_hard_pairs'), 'Reviewed training pairs are missing'
    return importlib.import_module('replycoach.reranker_hard_pairs')


def supplement(bundle, pair, label=0):
    return {'schema_version': 1, 'record_type': 'v6_reviewed_hard_pairs',
            'train_fingerprint': bundle['manifest']['train_fingerprint'],
            'validation_fingerprint': bundle['manifest']['validation_fingerprint'],
            'base_pair_sha256': bundle['manifest']['pair_sha256'],
            'source_review_sha256': 'a'*64, 'comparison_count': 1, 'skipped_count': 0,
            'pairs': [{**pair, 'label': label, 'provenance': 'human_reviewed_alternative'}]}


def test_reviewed_valid_alternative_replaces_a_weak_negative(inputs):
    b = build_training_bundle(inputs)
    old = copy.deepcopy(b)
    weak = next(p for p in b['train'] if not p['label'])
    rows, meta = api().apply_reviewed_pairs(b, supplement(b, weak, 1))
    assert b == old and len(rows) == len(b['train'])
    matches = [p for p in rows if p['item_id'] == weak['item_id'] and p['style'] == weak['style'] and p['reply'] == weak['reply']]
    assert len(matches) == 1 and matches[0]['label'] == 1
    assert matches[0]['provenance'] == 'human_reviewed_alternative'
    assert meta['weak_pairs_replaced'] == 1 and meta['weak_negative_labels_corrected'] == 1
    assert meta['training_pairs_sha256'] == sha256(pair_payload(rows)).hexdigest()


def test_reviewed_negative_is_added_without_changing_references_or_validation(inputs):
    b = build_training_bundle(inputs)
    own = b['train'][0]
    existing = {p['donor_id'] for p in b['train'] if p['item_id'] == own['item_id'] and p['style'] == own['style']}
    donor = next(p for p in b['train'] if p['label'] and p['style'] == own['style'] and p['item_id'] not in existing)
    pair = {**own, 'donor_id': donor['item_id'], 'reply': donor['reply'], 'label': 0}
    rows, meta = api().apply_reviewed_pairs(b, supplement(b, pair))
    assert len(rows) == len(b['train'])+1
    assert all(p in rows for p in b['train'] if p['label'])
    assert meta['reviewed_negative_count'] == 1
    assert all(set(p) == PAIR_KEYS for p in rows)


@pytest.mark.parametrize('change', ['validation_id','donor_validation','query','reply','fingerprint','bool_label','provenance','review_count'])
def test_foreign_or_invalid_reviewed_pairs_are_rejected(inputs, change):
    b = build_training_bundle(inputs)
    p = supplement(b, next(p for p in b['train'] if not p['label']))
    row = p['pairs'][0]
    if change == 'validation_id': row['item_id'] = b['manifest']['validation_ids'][0]
    elif change == 'donor_validation': row['donor_id'] = b['manifest']['validation_ids'][0]
    elif change == 'query': row['query'] += 'changed'
    elif change == 'reply': row['reply'] = 'new invented reply'
    elif change == 'fingerprint': p['train_fingerprint'] = 'wrong'
    elif change == 'bool_label': row['label'] = False
    elif change == 'provenance': row['provenance'] = 'pending'
    else: p['comparison_count'] = 0
    with pytest.raises(ValueError):
        api().apply_reviewed_pairs(b, p)


def test_conflicting_duplicates_and_private_extra_fields_are_rejected(inputs):
    b = build_training_bundle(inputs)
    p = supplement(b, next(p for p in b['train'] if not p['label']))
    p['pairs'].append({**p['pairs'][0], 'label': 1})
    p['comparison_count'] = 2
    with pytest.raises(ValueError): api().apply_reviewed_pairs(b, p)
    p = supplement(b, next(p for p in b['train'] if not p['label']))
    p['reviewer'] = 'PRIVATE_NAME'
    with pytest.raises(ValueError): api().apply_reviewed_pairs(b, p)
