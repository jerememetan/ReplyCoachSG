import copy
import importlib
import importlib.util
import json

import pytest

from replycoach.evaluation import load_evaluation_inputs


def api():
    assert importlib.util.find_spec('replycoach.reranker_training'), 'Training bundle is not implemented'
    return importlib.import_module('replycoach.reranker_training')


@pytest.fixture
def inputs(tmp_path):
    def item(i):
        return {'id': f'I{i}', 'split': 'dev', 'scenario': 'notes', 'intent': 'request',
                'session_memory': [{'speaker': 'peer', 'text': f'earlier {i}'},
                                   {'speaker': 'peer', 'text': f'also {i}'}],
                'message': {'text': f'incoming {i} 吃饭'},
                'replies': {'standard': f'standard {i}', 'local': f'local {i}'},
                'author': 'PRIVATE_AUTHOR', 'reviewer': 'PRIVATE_REVIEWER',
                'notes': 'PRIVATE_NOTES', 'status': 'completed', 'privacy_ok': True}
    train, val = tmp_path/'train.jsonl', tmp_path/'validation.jsonl'
    train.write_text('\n'.join(json.dumps(item(i)) for i in range(6)), encoding='utf-8')
    val.write_text('\n'.join(json.dumps(item(i)) for i in range(6, 8)), encoding='utf-8')
    return load_evaluation_inputs(val, train)


def test_training_pairs_preserve_speakers_style_and_strip_identity(inputs):
    bundle = api().build_training_bundle(inputs)
    assert len(bundle['train']) == 36 and len(bundle['validation']) == 12
    assert sum(p['label'] for p in bundle['train']) == 12
    pair = bundle['train'][0]
    assert 'Peer: earlier 0\nPeer: also 0' in pair['query']
    assert 'Requested reply style: STANDARD' in pair['query']
    assert '吃饭' in pair['query']
    assert 'PRIVATE_' not in json.dumps(bundle)


def test_weak_negatives_use_only_distinct_training_donors(inputs):
    bundle = api().build_training_bundle(inputs)
    train_ids = {i.id for i in inputs.bank.items}
    for split in ['train', 'validation']:
        for p in bundle[split]:
            if p['label'] == 0:
                assert p['donor_id'] in train_ids and p['donor_id'] != p['item_id']
                assert p['provenance'] == 'algorithmic_mismatch_weak'
                source = next(i for i in [*inputs.bank.items, *inputs.queries.items] if i.id == p['item_id'])
                assert p['reply'] not in [source.replies.standard, source.replies.local]
    assert 'false negatives' in bundle['manifest']['negative_policy']


def test_pair_sampling_is_reproducible_after_source_reordering(inputs):
    m = api()
    a = m.build_training_bundle(inputs)
    other = copy.deepcopy(inputs)
    other.bank.items.reverse()
    other.queries.items.reverse()
    b = m.build_training_bundle(other)
    assert a == b


def test_duplicate_replies_are_not_sampled_as_false_negatives(inputs):
    m = api()
    inputs.bank.items[1].replies.standard = inputs.bank.items[0].replies.standard
    bundle = m.build_training_bundle(inputs)
    for pair in bundle['train']:
        if pair['item_id'] == 'I0' and pair['style'] == 'STANDARD' and not pair['label']:
            assert pair['donor_id'] != 'I1'


def test_bundle_write_is_exclusive_and_checks_payload_hashes(inputs, tmp_path):
    m = api()
    bundle = m.build_training_bundle(inputs)
    out = tmp_path/'bundle'
    m.write_training_bundle(out, bundle)
    assert m.load_training_bundle(out) == bundle
    with pytest.raises(FileExistsError):
        m.write_training_bundle(out, bundle)
    p = out/'train_pairs.jsonl'
    p.write_text(p.read_text(encoding='utf-8').replace('standard 0', 'changed'), encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        m.load_training_bundle(out)


def test_test_items_and_overlapping_inputs_are_rejected(inputs):
    m = api()
    inputs.queries.items[0].split = 'test'
    with pytest.raises(ValueError, match='dev'):
        m.build_training_bundle(inputs)
    inputs.queries.items[0].split = 'dev'
    inputs.queries.items[0].id = inputs.bank.items[0].id
    with pytest.raises(ValueError, match='overlap'):
        m.build_training_bundle(inputs)


def test_malformed_pair_provenance_is_rejected(inputs):
    m = api()
    bundle = m.build_training_bundle(inputs)
    bad = next(p for p in bundle['train'] if not p['label'])
    bad['donor_id'] = inputs.queries.items[0].id
    with pytest.raises(ValueError, match='donor'):
        m.validate_training_bundle(bundle)
