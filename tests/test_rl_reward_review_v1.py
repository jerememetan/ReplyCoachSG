import importlib.util,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from prepare_rl_reward_review_v1 import prepare
from rl_reward_review import save_review

def test_partitions_are_group_disjoint_and_labels_are_proposals(tmp_path):
    packet=prepare(tmp_path/'data');groups={}
    for split in ('review','development','heldout'):
        rows=[json.loads(l) for l in (tmp_path/'data'/f'{split}.jsonl').read_text().splitlines()]
        assert len(rows)==24
        groups[split]={r['scenario_id'] for r in rows}
        assert len(groups[split])==6
        assert {r['category'] for r in rows}=={'time','availability','ownership'}
        assert all(r['proposed_reward'] is None for r in rows if r['proposed_verdict']=='uncertain')
    assert not groups['review']&groups['development']
    assert not groups['review']&groups['heldout']
    assert not groups['development']&groups['heldout']
    assert all(q['split']=='review' for q in packet['questions'])
    manifest=json.loads((tmp_path/'data/manifest.json').read_text())
    assert manifest['human_approved']==manifest['rl_updates']==0
    assert not manifest['scorer_implemented']
    assert prepare(tmp_path/'data')==packet

def test_review_persists_explicit_decisions_and_null_uncertainty(tmp_path):
    save_review(tmp_path,'case','acceptable','valid alternative')
    save_review(tmp_path,'case','uncertain','need more context')
    saved=json.loads((tmp_path/'human_reviews.json').read_text())
    assert saved['case']['reward'] is None and saved['case']['notes']=='need more context'
    with pytest.raises(ValueError):save_review(tmp_path,'other',None,'')
