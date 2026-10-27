"""Portable validation and application of explicitly reviewed training alternatives."""
from hashlib import sha256
import re

from replycoach.reranker_bundle import PAIR_KEYS, normalized, pair_payload, validate_training_bundle

SUPPLEMENT_KEYS = {'schema_version', 'record_type', 'train_fingerprint', 'validation_fingerprint',
                   'base_pair_sha256', 'source_review_sha256', 'comparison_count', 'skipped_count', 'pairs'}


def apply_reviewed_pairs(bundle, supplement):
    validate_training_bundle(bundle)
    manifest = bundle['manifest']
    if (set(supplement) != SUPPLEMENT_KEYS or supplement['schema_version'] != 1
            or supplement['record_type'] != 'v6_reviewed_hard_pairs'
            or any(supplement[k] != manifest[k] for k in ['train_fingerprint', 'validation_fingerprint'])
            or supplement['base_pair_sha256'] != manifest['pair_sha256']
            or not isinstance(supplement['source_review_sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', supplement['source_review_sha256'])
            or type(supplement['comparison_count']) is not int or supplement['comparison_count'] < 1
            or type(supplement['skipped_count']) is not int or supplement['skipped_count'] < 0
            or not isinstance(supplement['pairs'], list) or not supplement['pairs']
            or len(supplement['pairs'])+supplement['skipped_count'] != supplement['comparison_count']):
        raise ValueError('invalid reviewed supplement identity, source hashes or decision coverage')
    positives = {(p['item_id'], p['style']): p for p in bundle['train'] if p['label']}
    accepted = {}
    for p in supplement['pairs']:
        if (set(p) != PAIR_KEYS or type(p['label']) is not int or p['label'] not in {0, 1}
                or p['provenance'] != 'human_reviewed_alternative'):
            raise ValueError('reviewed pairs require an explicit human label and minimal schema')
        source = positives.get((p['item_id'], p['style']))
        donor = positives.get((p['donor_id'], p['style']))
        if (source is None or donor is None or p['item_id'] == p['donor_id']
                or p['query'] != source['query'] or p['reply'] != donor['reply']
                or normalized(p['reply']) in {normalized(positives[p['item_id'], s]['reply']) for s in ['STANDARD', 'LOCAL']}):
            raise ValueError('reviewed pairs must use unchanged training queries and distinct training reply donors')
        key = p['item_id'], p['style'], normalized(p['reply'])
        if key in accepted:
            raise ValueError('duplicate or conflicting reviewed labels')
        accepted[key] = dict(p)
    rows, replaced, corrected = [], 0, 0
    for p in bundle['train']:
        key = p['item_id'], p['style'], normalized(p['reply'])
        if key in accepted:
            if p['label']:
                raise ValueError('cannot replace a reviewed reference positive')
            replaced += 1
            corrected += accepted[key]['label'] == 1
        else:
            rows.append(dict(p))
    rows.extend(accepted[key] for key in sorted(accepted))
    metadata = {'method': 'human_reviewed_training_alternatives_v1',
                'source_review_sha256': supplement['source_review_sha256'],
                'comparison_count': supplement['comparison_count'], 'skipped_count': supplement['skipped_count'],
                'reviewed_positive_count': sum(p['label'] for p in accepted.values()),
                'reviewed_negative_count': sum(not p['label'] for p in accepted.values()),
                'weak_pairs_replaced': replaced, 'weak_negative_labels_corrected': corrected,
                'original_pair_count': len(bundle['train']), 'training_pair_count': len(rows),
                'training_pairs_sha256': sha256(pair_payload(rows)).hexdigest()}
    return rows, metadata
