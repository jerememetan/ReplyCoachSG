"""Parser fixtures only; these invented strings never enter training bundles."""
import importlib
import importlib.util
import json
import zipfile

import pytest
from test_reranker_training import inputs


def api():
    assert importlib.util.find_spec('replycoach.dailydialog_reranker'), 'DailyDialog adaptation is not implemented'
    return importlib.import_module('replycoach.dailydialog_reranker')


def dialogues(n=20):
    return [[f'hello topic {i}', f'answer topic {i}', f'follow up {i}', f'final answer {i}'] for i in range(n)]


def test_roles_follow_candidate_speaker_and_preserve_whole_turns():
    m = api()
    turns = ['first sentence. same speaker continues.', 'second turn', 'third turn', 'fourth turn']
    query = m.query_for_turn(turns, 3)
    assert 'Peer: first sentence. same speaker continues.' in query
    assert 'You: second turn' in query
    assert 'Incoming peer message: third turn' in query
    assert 'fourth turn' not in query
    assert 'Requested reply style: STANDARD' in query


def test_pairs_split_whole_dialogues_with_reproducible_training_only_donors():
    m = api()
    result = m.prepare_pairs(dialogues(), validation_fraction=.2, max_train_queries=12)
    again = m.prepare_pairs(list(reversed(dialogues())), validation_fraction=.2, max_train_queries=12)
    assert result == again
    train_ids = set(result['manifest']['train_ids'])
    val_ids = set(result['manifest']['validation_ids'])
    assert train_ids and val_ids and not train_ids & val_ids
    assert len(result['train']) == 36
    assert result['manifest']['max_train_queries'] == 12
    for split in ['train', 'validation']:
        for row in result[split]:
            assert row['item_id'] in (train_ids if split == 'train' else val_ids)
            if not row['label']:
                assert row['donor_id'] in train_ids
                assert row['donor_id'] != row['item_id']
                assert row['provenance'] == 'dailydialog_mismatch_weak'
            else:
                assert row['provenance'] == 'dailydialog_observed_next_turn_weak'
    assert 'false negatives' in result['manifest']['negative_policy']


def test_duplicates_and_project_overlap_are_removed_before_partitioning():
    m = api()
    source = dialogues()
    result = m.prepare_pairs(source + [source[0]], excluded_texts=['HELLO TOPIC 1!'], validation_fraction=.2)
    assert result['manifest']['duplicate_dialogues_removed'] == 1
    assert result['manifest']['project_overlap_dialogues_removed'] == 1
    assert len(result['manifest']['train_ids']) + len(result['manifest']['validation_ids']) == 19
    assert all('hello topic 1\n' not in p['query'] for split in ['train', 'validation'] for p in result[split])


def test_punctuation_only_turn_excludes_whole_dialogue_without_flipping_roles():
    result=api().prepare_pairs(dialogues()+[['question', '...', 'later answer']])
    assert result['manifest']['unusable_dialogues_removed']==1
    assert not any('later answer' in p['query'] or p['reply']=='later answer'
                   for s in ['train','validation'] for p in result[s])


def test_zip_parser_reads_only_turn_file_without_running_dataset_script(tmp_path):
    m = api()
    archive = tmp_path/'fixture.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('train/dialogues_train.txt', 'hello__eou__hi__eou__\n')
        z.writestr('train/dialogues_act_train.txt', '1 2\n')
    assert m.read_dialogues(archive) == [['hello', 'hi']]
    with pytest.raises(ValueError, match='hash'):
        m.download_archive(tmp_path/'bad.zip', fetch=lambda url: archive.read_bytes())


@pytest.mark.parametrize('turns', [[], ['one turn'], ['ok', ''], 'not a list'])
def test_invalid_dialogues_fail_instead_of_inventing_turns(turns):
    with pytest.raises(ValueError, match='dialogue'):
        api().prepare_pairs([turns])


def test_external_manifest_has_pair_hashes_and_truthful_speaker_assumption():
    result = api().prepare_pairs(dialogues())
    assert result['manifest']['speaker_policy'] == 'assumed_alternating_utterance_turns_not_verified_ids'
    assert len(result['manifest']['pair_sha256']['train']) == 64
    assert result['manifest']['source_split'] == 'train_only'
    assert result['manifest']['style_policy'] == 'STANDARD_only_no_invented_LOCAL_gold'


def test_pipeline_continues_stage_one_adapter_without_changing_project_bundle(inputs,tmp_path,monkeypatch):
    from test_reranker_fit import bundle_path
    from test_qwen_fit import tiny
    m=api()
    assert hasattr(m,'run_pipeline'), 'Two-stage orchestration is not implemented'
    bundle=bundle_path(inputs,tmp_path)
    original=(bundle/'train_pairs.jsonl').read_bytes()
    archive=tmp_path/'archive.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('train/dialogues_train.txt','\n'.join('__eou__'.join(d)+'__eou__' for d in dialogues()))
    monkeypatch.setattr(m,'download_archive',lambda *args,**kwargs:archive)
    result=m.run_pipeline(bundle,tmp_path/'two_stage',smoke=True,model_factory=tiny,
                          max_steps=1,max_train_queries=12,max_validation_queries=4)
    assert result['completed'] and not result['production_checkpoint']
    first=json.loads((tmp_path/'two_stage/stage1_dailydialog/best/qwen_training.json').read_text())
    final=json.loads((tmp_path/'two_stage/stage2_reviewed/best/qwen_training.json').read_text())
    assert final['initialization']['adapter_sha256']==first['adapter_sha256']
    assert final['validation_pairs_sha256']==json.loads((bundle/'manifest.json').read_text())['pair_sha256']['validation']
    assert (bundle/'train_pairs.jsonl').read_bytes()==original
