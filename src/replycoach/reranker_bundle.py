"""Portable V6 pair-bundle validation; also shipped with the private Kaggle trainer."""
from hashlib import sha256
import json
from pathlib import Path

BASE_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'
BASE_REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'
QUERY_FORMAT = 'speaker-conversation-style-v2'
PAIR_KEYS = {'item_id', 'donor_id', 'style', 'query', 'reply', 'label', 'provenance'}


def normalized(text):
    return ' '.join(text.split()).casefold()


def pair_payload(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n'
                   for row in rows).encode('utf-8')


def validate_training_bundle(bundle):
    manifest = bundle['manifest']
    train_ids, val_ids = set(manifest['train_ids']), set(manifest['validation_ids'])
    if (manifest['schema_version'] != 1 or manifest['query_format'] != QUERY_FORMAT
            or manifest['base_model'] != BASE_MODEL or manifest['base_revision'] != BASE_REVISION
            or not train_ids or not val_ids or train_ids & val_ids):
        raise ValueError('invalid or overlapping development bundle identity')
    positives = {}
    for split, ids in [('train', train_ids), ('validation', val_ids)]:
        rows = bundle[split]
        if not rows:
            raise ValueError('pair split must be nonempty')
        seen = set()
        for p in rows:
            if (set(p) != PAIR_KEYS or p['item_id'] not in ids or p['style'] not in {'STANDARD', 'LOCAL'}
                    or type(p['label']) is not int or p['label'] not in {0, 1}
                    or not isinstance(p['query'], str) or not p['query'].strip()
                    or not isinstance(p['reply'], str) or not p['reply'].strip()):
                raise ValueError('invalid pair schema, ID, text or label')
            key = (p['item_id'], p['style'], normalized(p['reply']))
            if key in seen:
                raise ValueError('duplicate pair')
            seen.add(key)
            if p['label'] == 1:
                cell = (p['item_id'], p['style'])
                if cell in positives or p['donor_id'] != p['item_id'] or p['provenance'] != 'human_reviewed_pair':
                    raise ValueError('invalid positive provenance')
                positives[cell] = p
        expected = {(item, style) for item in ids for style in ['STANDARD', 'LOCAL']}
        if {cell for cell in positives if cell[0] in ids} != expected:
            raise ValueError('positive coverage must include both styles for every item')
    for split in ['train', 'validation']:
        for p in bundle[split]:
            own = positives[p['item_id'], p['style']]
            if p['query'] != own['query']:
                raise ValueError('one item/style must share the same query')
            if not p['label']:
                known = {normalized(positives[p['item_id'], s]['reply']) for s in ['STANDARD', 'LOCAL']}
                donor = positives.get((p['donor_id'], p['style']))
                if (p['donor_id'] not in train_ids or p['donor_id'] == p['item_id'] or donor is None
                        or normalized(p['reply']) != normalized(donor['reply'])
                        or normalized(p['reply']) in known or p['provenance'] != 'algorithmic_mismatch_weak'):
                    raise ValueError('invalid weak-negative donor or provenance')
    return bundle


def load_training_bundle(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    bundle = {'manifest': manifest}
    for split in ['train', 'validation']:
        raw = (path/f'{split}_pairs.jsonl').read_bytes()
        if sha256(raw).hexdigest() != manifest['pair_sha256'][split]:
            raise ValueError(f'{split} pair file hash mismatch')
        bundle[split] = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    return validate_training_bundle(bundle)


def write_training_bundle(path, bundle):
    validate_training_bundle(bundle)
    path = Path(path)
    if 'data' in path.resolve().parts and (path.resolve() == Path(__file__).resolve().parents[2]/'data'
                                         or (Path(__file__).resolve().parents[2]/'data') in path.resolve().parents):
        raise ValueError('write bundles outside the source data directory')
    path.mkdir(parents=True, exist_ok=False)
    for split in ['train', 'validation']:
        raw = pair_payload(bundle[split])
        if sha256(raw).hexdigest() != bundle['manifest']['pair_sha256'][split]:
            raise ValueError('pair hash mismatch before write')
        (path/f'{split}_pairs.jsonl').write_bytes(raw)
    (path/'manifest.json').write_text(json.dumps(bundle['manifest'], ensure_ascii=False,
                                               sort_keys=True, indent=2), encoding='utf-8')
