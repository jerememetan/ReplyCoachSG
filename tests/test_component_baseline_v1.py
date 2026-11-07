import importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('component_baseline',ROOT/'scripts/run_component_baseline_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_generation_payload_excludes_references_and_labels():
 row=dict(item_id='secret',category='secret',standard='gold',expectation='gold',context=['I have your book.','Thanks.'],speakers=['user','peer'],message='Return it tomorrow?')
 p=m.conversation_payload(row)
 assert set(p)=={'conversation','incoming_message'}
 assert p['conversation'][0]['speaker']=='user' and p['incoming_message']['speaker']=='peer'
 assert 'gold' not in json.dumps(p) and 'secret' not in json.dumps(p)
 with pytest.raises(ValueError):m.conversation_payload({**row,'speakers':[]})
def test_rewrite_uses_exact_anchor_and_no_labels():
 anchor="I haven't booked it yet."
 p=m.rewrite_payload(anchor,[dict(item_id='hidden',standard='Okay.',local='ok can',expectation='hidden')])
 assert p['standard_draft']==anchor
 assert 'hidden' not in json.dumps(p)
 with pytest.raises(ValueError):m.rewrite_payload('',[])
def test_frozen_cases_roles_and_unique_ids():
 folder=ROOT/'data/evaluation/component_baseline_v1';inputs=json.loads((folder/'inputs.json').read_text());refs=json.loads((folder/'reference_anchors.json').read_text())
 assert len(inputs)==24 and {r['item_id'] for r in inputs}=={r['item_id'] for r in refs}
 assert len({r['message'] for r in inputs})==24
 assert all(r['speakers']==['user','peer'] for r in inputs)
 assert all('standard' not in r and 'expectation' not in r for r in inputs)

def test_mt5_inference_supplies_empty_targets_not_references():
 import sys
 sys.path.insert(0,str(ROOT/'scripts'))
 from run_component_mt5_v1 import inference_rows
 r=dict(context=[],speakers=[],message='Hi',standard='hidden reference')
 output=inference_rows([r])
 assert output[0]['response']=='' and 'response' not in r
 assert output[0]['message']=='Hi'
