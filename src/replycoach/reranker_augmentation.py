"""Portable, seeded input-only augmentation; inherited labels remain weak supervision."""
from hashlib import sha256
import json
import math
import random
import re

from replycoach.reranker_bundle import pair_payload, validate_training_bundle


def augment_training_pairs(bundle, mapping, *, copies=3, probability=.35, seed=425):
    validate_training_bundle(bundle)
    if (type(copies) is not int or not 0 <= copies <= 20 or type(seed) is not int
            or type(probability) not in {float, int} or not math.isfinite(probability)
            or not 0 < probability <= 1):
        raise ValueError('invalid augmentation copies, probability or seed')
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('a nonempty abbreviation mapping is required')
    for word, forms in mapping.items():
        if (not isinstance(word, str) or not re.fullmatch('[a-z]+', word)
                or not isinstance(forms, list) or not forms
                or any(not isinstance(s, str) or not re.fullmatch('[a-z]+', s) or s == word for s in forms)):
            raise ValueError('mapping must contain lowercase words and nonempty short-form lists')
    mapping = {w: sorted(set(forms)) for w, forms in sorted(mapping.items())}
    pattern = re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(w) for w in mapping)+r')(?!\w)', re.I)
    ordered = sorted(bundle['train'], key=lambda p:
                     (p['item_id'], p['style'], -p['label'], p['donor_id'], p['reply'])) if copies else bundle['train']
    rows = [dict(p) for p in ordered]
    originals = list(rows)
    groups = {}
    for p in originals:
        groups.setdefault(p['item_id'], []).append(p)
    contexts = 0
    for item_id, group in groups.items():
        bodies = set()
        for p in group:
            suffix = f"\nRequested reply style: {p['style']}\nUser reply:"
            if not p['query'].startswith('Conversation (oldest first):\n') or not p['query'].endswith(suffix):
                raise ValueError('augmentation requires the protected style-aware conversation format')
            bodies.add(p['query'][len('Conversation (oldest first):\n'):-len(suffix)])
        if len(bodies) != 1:
            raise ValueError('both styles must share the same original conversation')
        body = bodies.pop()
        rng = random.Random(int(sha256(f'{seed}|{item_id}'.encode()).hexdigest(), 16))
        seen = {body}
        variants = []
        for _ in range(max(30, copies*20) if copies else 0):
            if len(variants) >= copies:
                break
            lines = []
            for line in body.split('\n'):
                role = re.match(r'^(?:You: |Peer: |Other: |Unknown: |Incoming peer message: )', line)
                prefix = role.group() if role else ''
                text = line[len(prefix):]
                text = pattern.sub(lambda match: rng.choice(mapping[match.group().lower()])
                                   if rng.random() < probability else match.group(), text)
                lines.append(prefix+text)
            changed = '\n'.join(lines)
            if changed not in seen:
                seen.add(changed)
                variants.append(changed)
        contexts += len(variants)
        for number, changed in enumerate(variants, 1):
            for p in group:
                query = ('Conversation (oldest first):\n'+changed
                         +f"\nRequested reply style: {p['style']}\nUser reply:")
                rows.append({**p, 'query': query, 'provenance': 'augmented_inherited_label',
                             'augmentation': {'parent_item_id': item_id, 'variant': number,
                                              'source_provenance': p['provenance']}})
    metadata = {'method': 'text_abbreviation_v1', 'copies_requested': copies,
                'replacement_probability': probability, 'seed': seed,
                'mapping_sha256': sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(),
                'original_pair_count': len(originals), 'expanded_pair_count': len(rows),
                'augmented_pair_count': len(rows)-len(originals), 'generated_context_count': contexts,
                'underlying_conversations': len(groups),
                'expanded_pairs_sha256': sha256(pair_payload(rows)).hexdigest(),
                'label_policy': 'Inherited labels; augmented positives are not separately reviewed; weak negatives may be valid replies'}
    return rows, metadata
