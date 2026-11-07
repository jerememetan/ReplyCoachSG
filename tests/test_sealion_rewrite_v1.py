import json
from pathlib import Path
import pytest
from replycoach.generate.sealion_rewrite_v1 import rewrite_source,encode_rewrite
ROOT=Path(__file__).resolve().parents[1]
class Tokenizer:
    eos_token_id=1
    def encode(self,text,add_special_tokens=False):return [ord(c)+2 for c in text]

def row():return dict(item_id='r',context=['You have my cable.'],speakers=['user'],message='Can I return it Monday?',standard_anchor='Yes, please return my cable Monday.',response='Can, Monday return me ah.',diagnostic_anchor='HIDDEN DIAGNOSTIC',direct_local='HIDDEN BASELINE')
def test_prompt_only_contains_explicit_anchor_and_conversation():
    r=row();prompt=rewrite_source(r,'ACTUAL ENGLISH ANCHOR')
    assert 'ACTUAL ENGLISH ANCHOR' in prompt and 'Friend: Can I return it Monday?' in prompt
    assert r['response'] not in prompt and r['standard_anchor'] not in prompt and 'HIDDEN' not in prompt

def test_anchor_and_context_are_masked_and_real_eos_is_retained():
    r=row();tok=Tokenizer();e=encode_rewrite(r,tok,1000);n=e['prompt_tokens']
    assert e['labels'][:n]==[-100]*n
    assert e['labels'][n:]==tok.encode(r['response'])+[tok.eos_token_id]
    assert e['labels'][-1]==1

def test_empty_anchor_and_token_overflow_are_errors():
    with pytest.raises(ValueError,match='anchor'):rewrite_source(row(),' ')
    with pytest.raises(ValueError,match='truncation'):encode_rewrite(row(),Tokenizer(),10)

def test_audit_excludes_conflicting_pairs_without_editing_originals():
    audit=json.loads((ROOT/'data/training_expansion/sealion_rewrite_v1/pair_audit.json').read_text());lookup={x['id']:x for x in audit['items']}
    assert not lookup['ITEM-B-029']['eligible'] and not lookup['ITEM-C-047']['eligible']
    assert sum(x['eligible'] for x in audit['items'] if x['split']=='train')==745
    assert sum(x['eligible'] for x in audit['items'] if x['split']=='validation')==42
    assert not any(x['human_gold'] for x in audit['items'])

def test_evaluation_actual_anchors_exactly_match_fixed_english_baseline():
    rows=json.loads((ROOT/'data/evaluation/sealion_rewrite_v1/inputs.json').read_text());baseline=json.loads((ROOT/'artifacts/sealion_corrected_v1/s425.json').read_text());b={x['item_id']:x for cases in baseline['outputs'].values() for x in cases}
    assert len(rows)==82 and sum('diagnostic_anchor' in x for x in rows)==32
    for r in rows:
        assert r['actual_anchor']==b[r['item_id']]['styles']['STANDARD']['reply']
        assert r['direct_local']==b[r['item_id']]['styles']['LOCAL']['reply']
