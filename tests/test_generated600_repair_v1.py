import hashlib,importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('repair600',ROOT/'scripts/build_generated600_repair_v1.py');repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
FOLDER=ROOT/'data/training_expansion/generated600_repair_v1'
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def test_team_authored_records_are_preserved_byte_for_byte():
    original=(ROOT/'data/dev/train.jsonl').read_bytes();combined=(FOLDER/'combined_train.jsonl').read_bytes()
    assert combined.startswith(original)
    assert rows(FOLDER/'combined_train.jsonl')[:190]==rows(ROOT/'data/dev/train.jsonl')
    assert len(rows(FOLDER/'combined_train.jsonl'))==790

def test_audit_is_complete_replayable_and_does_not_invent_human_ratings():
    team={r['id'] for r in rows(ROOT/'data/dev/train.jsonl')};source=[r for r in rows(ROOT/'data/training_expansion/blanket600_v1/expanded.jsonl') if r['id'] not in team]
    updated=rows(FOLDER/'generated_corrected.jsonl');audit=rows(FOLDER/'audit.jsonl');patches=json.loads((FOLDER/'patches.json').read_text())
    assert len(source)==len(updated)==len(audit)==600
    assert len({r['id'] for r in updated})==600
    for n,(old,new,entry) in enumerate(zip(source,updated,audit,strict=True),1):
        expected=repair.apply_patch(old,patches[str(n)]) if str(n) in patches else old
        assert {k:v for k,v in new.items() if k!='assistant_revision'}==expected
        assert old['id']==new['id']==entry['id']
        assert entry['human_per_item_rating'] is False and new['assistant_revision']['human_per_item_rating'] is False
        actual={k:{'before':old[k],'after':new[k]} for k in ['session_memory','message','replies'] if old[k]!=new[k]}
        assert entry['changes']==actual
    assert sum(bool(x['changes']) for x in audit)==515

def test_dataset_has_no_validation_ids_or_exact_input_leakage_and_hashes_match():
    combined=rows(FOLDER/'combined_train.jsonl');validation=rows(ROOT/'data/dev/validation.jsonl')
    assert not {r['id'] for r in combined}&{r['id'] for r in validation}
    assert not {repair.signature(r) for r in combined}&{repair.signature(r) for r in validation}
    manifest=json.loads((FOLDER/'manifest.json').read_text())
    for p,h in manifest['files_sha256'].items():assert hashlib.sha256((FOLDER/p).read_bytes()).hexdigest()==h
    assert manifest['sources']['team_sha256']==hashlib.sha256((ROOT/'data/dev/train.jsonl').read_bytes()).hexdigest()
    assert not manifest['training_started']

def test_team_patch_and_mismatched_role_edits_are_rejected():
    original=rows(ROOT/'data/dev/train.jsonl')[0]
    with pytest.raises(ValueError,match='Team-authored'):repair.apply_patch(original,dict(id=original['id'],local='Changed',reason='test'))
    generated=rows(FOLDER/'generated_corrected.jsonl')[0]
    with pytest.raises(ValueError,match='Role count'):repair.apply_patch(generated,dict(id=generated['id'],context_speakers=[],reason='test'))
