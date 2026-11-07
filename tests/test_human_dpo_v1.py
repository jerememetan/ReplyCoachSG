import importlib.util,json
from pathlib import Path
import pytest
import torch
from replycoach.generate.human_dpo_v1 import fit
from test_train_v5 import tiny_model


def test_export_groups_conversations_and_excludes_ambiguous_labels(tmp_path):
    spec=importlib.util.spec_from_file_location('prepare_human_dpo',Path(__file__).parents[1]/'scripts/prepare_human_dpo_v1.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    root=Path(__file__).parents[1]
    m=mod.prepare(root/'data/human_review/model_ab_v1',root/'data/dev/validation.jsonl',tmp_path/'prefs')
    train=[json.loads(l) for l in (tmp_path/'prefs/train.jsonl').read_text().splitlines()]
    val=[json.loads(l) for l in (tmp_path/'prefs/validation.jsonl').read_text().splitlines()]
    assert not {r['item_id'] for r in train}&{r['item_id'] for r in val}
    assert m['excluded']['tie']==27 and m['excluded']['both_bad']==29
    assert m['raw_decisive']==94 and m['unique_pairs']==84
    directions={(r['item_id'],r['chosen'],r['rejected']) for r in train+val}
    assert all((i,b,a) not in directions for i,a,b in directions)


def test_small_dpo_pilot_records_baseline_and_saves_selected_checkpoint(tmp_path):
    torch.set_num_threads(1)
    base=tiny_model(tmp_path/'base',['Can you help me?','Yes I can help.','No I cannot.'])
    row=dict(item_id='train1',context=[],speakers=[],message='Can you help me?',chosen='Yes I can help.',rejected='No I cannot.')
    val={**row,'item_id':'val1'}
    audit=[dict(item_id='val1',style='STANDARD',context=[],speakers=[],message=row['message'],response=row['chosen'])]
    report=fit(base,[row],[val],audit,tmp_path/'run',device='cpu',max_epochs=1)
    assert report['status']=='complete' and report['optimizer_steps']==1
    assert report['history'][0]['validation']['dpo_loss']==pytest.approx(.693147,abs=1e-5)
    assert len(report['baseline_generations'])==len(report['dpo_generations'])==1
    assert (tmp_path/'run/best/v5_input.json').exists()
    with pytest.raises(ValueError,match='leakage'):fit(base,[row],[row],[],tmp_path/'bad',device='cpu')
