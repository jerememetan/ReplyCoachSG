import copy,json,sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from replycoach.generate.native_rewrite_v2 import source,StyleExamples,MODES
from replycoach.generate.native_rewrite_train_v2 import fit
from test_train_v5 import tiny_model


def rows():
    def row(id,group,split,standard,local):return dict(item_id=id,group_id=group,split=split,context=['Private context'],speakers=['user'],message='Private incoming',standard=standard,standard_variants=[standard],local=local)
    train=[row('ITEM-A-1','A','train','I will come.','I come lah.'),row('NATIVE-PUB-train-1','pub1','train','I cannot join.','Cannot join lor.')]
    val=[row('ITEM-E-1','E','validation','I will stay.','I stay ah.'),row('NATIVE-PUB-validation-1','pubval','validation','I am coming.','I coming liao.')]
    anchors=[{**r,'item_id':'FRESH-'+r['item_id'],'group_id':'FRESH-'+r['group_id'],'split':'test'} for r in val]
    return train,val,anchors


def test_context_and_reference_target_do_not_leak_into_anchor_only_or_exemplar_source():
    train,val,anchors=rows();r=val[0];pool=StyleExamples(train)
    text,_=source(r,'anchor_only');assert r['standard'] in text
    assert r['context'][0] not in text and r['message'] not in text and r['local'] not in text
    altered={**r,'local':'SECRET VALIDATION TARGET'}
    assert source(r,'exemplar',pool)==source(altered,'exemplar',pool)
    assert 'SECRET' not in source(altered,'exemplar',pool)[0]
    assert source(train[0],'exemplar',pool)[1]!=train[0]['item_id']
    with pytest.raises(ValueError):StyleExamples(train+val)


def test_full_context_keeps_exact_anchor_and_explicit_roles():
    r=rows()[0][0];text,_=source(r,'full_context')
    assert 'You: Private context' in text and 'Friend: Private incoming' in text
    assert 'Standard draft: I will come.' in text


@pytest.mark.parametrize('mode',MODES)
def test_real_cpu_training_selects_checkpoint_and_preserves_actual_anchors(tmp_path,mode):
    torch.set_num_threads(1);train,val,anchors=rows();pool=StyleExamples(train)
    texts=[source(r,m,pool)[0] for r in train+val+anchors for m in MODES]+[r['local'] for r in train+val]
    base=tiny_model(tmp_path/'base',texts)
    result=fit(base,train,val,anchors,tmp_path/'run',mode,device='cpu',batch_size=2,max_epochs=1)
    assert result['status']=='complete' and result['optimizer_steps']==1
    assert all('train_token_loss' in e and 'validation_macro' in e for e in result['history'])
    assert len(result['fresh_outputs'])==2 and all(r['local_conditioned_on']==r['standard'] for r in result['fresh_outputs'])
    assert (tmp_path/'run/best/native_rewrite_config.json').exists()


def test_actual_native_targets_remain_verbatim_and_groups_are_disjoint():
    root=Path(__file__).parents[1];folder=root/'data/public/native_rewrite_v2'
    if not (folder/'train.jsonl').exists():pytest.skip('Private derived corpus not distributed with repository')
    allrows=[json.loads(l) for name in ('train','validation') for l in (folder/f'{name}.jsonl').read_text().splitlines()]
    originals={r['id']:r['replies']['local'] for r in map(json.loads,(root/'data/dev/train.jsonl').read_text().splitlines())}
    native={r['item_id']:r['local'] for r in json.loads((folder/'teacher_normalizations.json').read_text())['items']}
    for r in allrows:assert r['local']==(originals[r['item_id']] if r['item_id'].startswith('ITEM-') else native[r['item_id']])
    assert len(allrows)==283
    assert not {r['group_id'] for r in allrows if r['split']=='train'}&{r['group_id'] for r in allrows if r['split']=='validation'}
