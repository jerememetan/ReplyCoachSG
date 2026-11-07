import copy,importlib.util,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_sealion_expanded_v1 import expanded_rows

def data():
    def read(p):return [json.loads(l) for l in (ROOT/p).read_text().splitlines()]
    return read('data/training_expansion/blanket600_v1/expanded.jsonl'),read('data/dev/train.jsonl'),read('data/dev/validation.jsonl')
def test_preserved_arm_restores_original_targets_without_changing_additions():
    expanded,original,val=data();saved=expanded_rows(expanded,original,val,'saved');preserved=expanded_rows(expanded,original,val,'preserved');old={r['id']:r for r in original}
    assert len(saved)==len(preserved)==1580
    for a,b in zip(saved,preserved,strict=True):
        assert {k:v for k,v in a.items() if k!='response'}=={k:v for k,v in b.items() if k!='response'}
        if a['item_id'] in old:assert b['response']==old[a['item_id']]['replies'][a['style'].lower()]
        else:assert a==b
    assert sum(a['response']!=b['response'] for a,b in zip(saved,preserved))==204
    assert expanded==data()[0]
def test_validation_id_overlap_is_rejected():
    expanded,original,val=data();val.append({'id':expanded[-1]['id']})
    with pytest.raises(ValueError,match='split overlap'):expanded_rows(expanded,original,val,'saved')
def test_original_context_changes_are_rejected():
    expanded,original,val=data();expanded[0]['message']['text']='changed input'
    with pytest.raises(ValueError,match='Original input changed'):expanded_rows(expanded,original,val,'preserved')

def test_corrected_source_preserves_team_bytes_and_is_selected_explicitly(monkeypatch,tmp_path):
    import run_sealion_expanded_v1 as runner
    expanded,original,val=data()
    recipe=dict(training_file='data/training_expansion/generated600_repair_v1/combined_train.jsonl',require_original_team_exact=True)
    checked=runner.checked_training_rows(recipe,original,val,'preserved')
    assert len(checked)==1580
    actual=[json.loads(l) for l in (ROOT/recipe['training_file']).read_text().splitlines()]
    expected=runner.expanded_rows(actual,original,val,'preserved')
    assert checked==expected
    (tmp_path/'data/dev').mkdir(parents=True)
    (tmp_path/'data/dev/train.jsonl').write_bytes((ROOT/'data/dev/train.jsonl').read_bytes())
    actual[0]['replies']['local']='Unwanted team rewrite'
    (tmp_path/'combined.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in actual))
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='byte-for-byte'):
        runner.checked_training_rows(dict(training_file='combined.jsonl',require_original_team_exact=True),original,val,'preserved')
