import json,runpy
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
RUN=runpy.run_path(str(ROOT/'scripts/run_mt5_target_repair_v1.py'))


def test_only_five_response_targets_change():
    a,b=RUN['training_rows']('control'),RUN['training_rows']('repair')
    expected=json.loads((ROOT/'data/training_expansion/mt5_grounding_repair_v1/manifest.json').read_text())['changed_target_ids']
    assert set(RUN['verify_controls'](a,b,expected))==set(expected)
    tampered=[dict(r) for r in b];tampered[0]['message']='changed context'
    with pytest.raises(ValueError,match='Unmatched'):RUN['verify_controls'](a,tampered,expected)


def test_source_and_original_validation_do_not_expose_targets():
    row=dict(context=['I have your pen.','Thanks.'],speakers=['user','peer'],message='Can you return it?',response='SECRET',reply_acts=['SECRET'],expectation='SECRET')
    text=RUN['source'](row)
    assert 'SECRET' not in text and 'Friend:' in text and 'You:' in text
    val=RUN['original_validation']()
    assert len(val)==50 and not {r['item_id'] for r in val}&{r['item_id'] for r in RUN['training_rows']('control')}


def test_fresh_inputs_have_no_references_and_no_exact_training_overlap():
    folder=ROOT/'data/evaluation/mt5_target_repair_v1'
    fresh=json.loads((folder/'inputs.json').read_text())
    assert len(fresh)==12 and len({r['item_id'] for r in fresh})==12
    assert all(set(r)=={'item_id','context','speakers','message'} for r in fresh)
    old={RUN['source'](r) for r in RUN['training_rows']('control')+RUN['original_validation']()}
    assert not {RUN['source'](r) for r in fresh}&old


def test_rewrite_inference_always_has_empty_response_without_mutating_inputs():
    rows=[{'item_id':'fresh','message':'Hi','context':[],'speakers':[]},
          {'item_id':'validation','message':'Hi','context':[],'speakers':[],'response':'SECRET_REFERENCE'}]
    encoded=RUN['rewrite_inference_rows'](rows)
    assert all(row['response']=='' for row in encoded)
    assert 'response' not in rows[0] and rows[1]['response']=='SECRET_REFERENCE'
