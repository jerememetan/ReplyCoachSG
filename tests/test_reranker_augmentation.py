import copy
import importlib
import importlib.util
import json

import pytest

from test_reranker_training import inputs
from replycoach.reranker_training import build_training_bundle


def api():
    assert importlib.util.find_spec('replycoach.reranker_augmentation'), 'Abbreviation augmentation is missing'
    return importlib.import_module('replycoach.reranker_augmentation')


@pytest.fixture
def bundle(inputs):
    for item in inputs.bank.items:
        item.message.text = 'Can you confirm tomorrow at 5? 吃饭'
        item.session_memory[0].text = 'You remember the library and school'
    return build_training_bundle(inputs)


MAPPING = {'you': ['u'], 'confirm': ['cfm'], 'tomorrow': ['tmr', 'tmrw'],
           'remember': ['rmb'], 'library': ['lib'], 'school': ['sch']}


def test_variants_preserve_labels_replies_roles_style_and_validation(bundle):
    original = copy.deepcopy(bundle)
    rows, meta = api().augment_training_pairs(bundle, MAPPING, copies=3, probability=.5, seed=425)
    extra = [p for p in rows if 'augmentation' in p]
    assert extra and len(rows) <= len(bundle['train'])*4
    assert bundle == original
    assert all(p in rows for p in bundle['train'])
    by_item = {}
    for p in extra:
        source = next(b for b in bundle['train'] if b['item_id'] == p['item_id']
                      and b['style'] == p['style'] and b['donor_id'] == p['donor_id'])
        assert p['reply'] == source['reply'] and p['label'] == source['label']
        assert p['provenance'] == 'augmented_inherited_label'
        assert p['augmentation']['source_provenance'] == source['provenance']
        assert p['query'] != source['query']
        assert '\nPeer: ' in p['query'] and '\nYou:' not in p['query']
        assert p['query'].endswith(f"Requested reply style: {p['style']}\nUser reply:")
        assert '5? 吃饭' in p['query']
        body = p['query'].split('\nRequested reply style:')[0]
        key = (p['item_id'], p['augmentation']['variant'])
        by_item.setdefault(key, set()).add(body)
    assert all(len(contexts) == 1 for contexts in by_item.values())
    assert meta['original_pair_count'] == len(bundle['train'])
    assert meta['expanded_pair_count'] == len(rows)
    assert meta['augmented_pair_count'] == len(extra)
    assert meta['underlying_conversations'] == 6
    assert meta['expanded_pairs_sha256'] and meta['mapping_sha256']
    assert {p['item_id'] for p in rows} == set(bundle['manifest']['train_ids'])


def test_randomness_is_seeded_reproducible_and_order_independent(bundle):
    m = api()
    a = m.augment_training_pairs(bundle, MAPPING, seed=425)
    other = copy.deepcopy(bundle)
    other['train'].reverse()
    assert a == m.augment_training_pairs(other, dict(reversed(list(MAPPING.items()))), seed=425)
    b = m.augment_training_pairs(bundle, MAPPING, seed=426)
    assert a[0] != b[0]


def test_noop_and_duplicate_contexts_are_not_padded(bundle):
    m = api()
    rows, meta = m.augment_training_pairs(bundle, {'missing': ['miss']})
    assert not any('augmentation' in p for p in rows)
    assert meta['augmented_pair_count'] == 0
    rows, meta = m.augment_training_pairs(bundle, {'confirm': ['cfm']}, copies=3, probability=1.)
    assert len(rows) == 2*len(bundle['train'])  # one possible changed context per item
    assert meta['generated_context_count'] == 6


def test_zero_copies_preserves_the_exact_baseline_training_order(bundle):
    rows, meta = api().augment_training_pairs(bundle, MAPPING, copies=0)
    assert rows == bundle['train']
    assert meta['augmented_pair_count'] == 0


def test_whole_words_and_speaker_labels_are_protected(bundle):
    m = api()
    from replycoach.reranker_bundle import pair_payload
    from hashlib import sha256
    for p in bundle['train']:
        p['query'] = p['query'].replace('Can you confirm tomorrow at 5? 吃饭',
                                       'your yourself you YOU you2 xyou tomorrowish tomorrow at 5? 吃饭')
        p['query'] = p['query'].replace('Peer: You remember', 'You: You remember')
    bundle['manifest']['pair_sha256']['train'] = sha256(pair_payload(bundle['train'])).hexdigest()
    rows, _ = m.augment_training_pairs(bundle, {'you': ['u'], 'tomorrow': ['tmr']}, probability=1.)
    for p in rows:
        if 'augmentation' in p:
            assert 'your yourself u u you2 xyou tomorrowish tmr at 5? 吃饭' in p['query']
            assert '\nYou: u remember' in p['query'] and '\nu:' not in p['query']


@pytest.mark.parametrize('settings', [{'copies': -1}, {'copies': True}, {'copies': 21},
                                     {'probability': 0}, {'probability': float('nan')},
                                     {'probability': 1.1}, {'seed': True}])
def test_invalid_settings_are_rejected(bundle, settings):
    with pytest.raises(ValueError):
        api().augment_training_pairs(bundle, MAPPING, **settings)


def test_mapping_is_conservative_snapshot_of_existing_abbr():
    mapping, source = importlib.import_module('replycoach.reranker_packaging').abbreviation_snapshot()
    assert mapping['confirm'] == ['cfm'] and mapping['you'] == ['u']
    assert set(mapping['tomorrow']) == {'tmr', 'tmrw'}
    assert not {'gg', 'ur', 'thr', 'gna', 'anot'} & {s for variants in mapping.values() for s in variants}
    assert source['path'] == 'scripts/data-split/near_dups.py'
    assert len(source['sha256']) == 64
    json.dumps(mapping, allow_nan=False)
