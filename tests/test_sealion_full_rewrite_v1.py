import json
from copy import deepcopy
from pathlib import Path
import pytest
from replycoach.generate.sealion_full_rewrite_v1 import frozen_inputs
ROOT=Path(__file__).resolve().parents[1]

def data():
    old=json.loads((ROOT/'data/evaluation/sealion_rewrite_v1/inputs.json').read_text())
    baseline=json.loads((ROOT/'artifacts/sealion_full_v1/results.json').read_text())
    return old,baseline

def test_new_renderer_uses_full_model_english_without_changing_conversations_or_diagnostics():
    old,baseline=data();snapshot=deepcopy(old);new=frozen_inputs(old,baseline)
    assert old==snapshot and len(new)==82
    mapping={r['item_id']:r for cases in baseline['outputs'].values() for r in cases}
    for a,b in zip(old,new,strict=True):
        for field in ['context','speakers','message']:
            assert a[field]==b[field]
        assert a.get('diagnostic_anchor')==b.get('diagnostic_anchor')
        direct=mapping[b['item_id']]['styles']
        assert b['actual_anchor']==direct['STANDARD']['reply']
        assert b['direct_local']==direct['LOCAL']['reply']
    assert sum(a['actual_anchor']!=b['actual_anchor'] for a,b in zip(old,new,strict=True))>0

def test_wrong_or_incomplete_baseline_is_rejected():
    old,baseline=data();baseline['status']='running'
    with pytest.raises(ValueError,match='complete'):frozen_inputs(old,baseline)
    old,baseline=data();old.pop()
    with pytest.raises(ValueError,match='coverage'):frozen_inputs(old,baseline)
