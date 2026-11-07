import importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('diagnostic_v3',ROOT/'scripts/run_rewrite_diagnostic_v3.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def rows(name):return [json.loads(s) for s in (ROOT/f'data/training_expansion/rewrite_diagnostic_v3/{name}.jsonl').read_text().splitlines()]
def test_original_local_targets_and_split_guards():
 train,val=rows('train'),rows('validation');module.validate(train,val)
 original={r['item_id']:r['local'] for r in json.loads((ROOT/'data/public/native_rewrite_v2/original_repaired.json').read_text())}
 assert all(original[r['item_id']]==r['local'] for r in train+val)
 altered=[dict(r) for r in val];altered[0]['group_id']=train[0]['group_id']
 with pytest.raises(ValueError,match='Author overlap'):module.validate(train,altered)
 altered=[dict(r) for r in val];altered[0]['local']=train[0]['local']
 with pytest.raises(ValueError,match='Target overlap'):module.validate(train,altered)
def test_inputs_do_not_read_targets():
 from replycoach.generate.native_rewrite_v2 import source
 r=rows('train')[0];assert source(r,'anchor_only')==source({**r,'local':'different target','message':'different context'},'anchor_only')
