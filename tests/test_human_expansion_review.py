import csv,importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('human_expansion_review',Path(__file__).resolve().parents[1]/'scripts/human_expansion_review.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)

def setup(tmp_path):
    source=tmp_path/'source.jsonl'
    items=[dict(id=f'SYN-H{i}',session_memory=[],message=f'Could you send file {i}?',replies={'standard':'I will send it.','local':'Can, I send you.'},scenario_group='sending',provenance={'generator':'assistant','prompt_version':'test','batch_id':'test'}) for i in range(4)]
    source.write_text(''.join(json.dumps(c)+'\n' for c in items))
    out=tmp_path/'review';review.prepare(source,out,4)
    file=out/'batch_01.csv'
    with file.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    return out,file,rows

def test_only_explicit_good_exports_and_corrections_are_preserved(tmp_path):
    root,file,rows=setup(tmp_path)
    rows[0]['decision']='Good';rows[0]['corrected_standard']='I will email it shortly.';rows[0]['corrected_local']='Can, later I email you.'
    rows[1]['decision']='bad';rows[2]['decision']='unsure';rows[3]['corrected_standard']='A correction without approval.'
    review.write_csv(file,rows)
    out=tmp_path/'export';review.import_reviews(root/'manifest.json',[file],'Test reviewer',out)
    approved=[json.loads(l) for l in (out/'approved.jsonl').read_text().splitlines()]
    assert len(approved)==1 and approved[0]['id']==rows[0]['id']
    assert approved[0]['replies']['standard']=='I will email it shortly.'
    assert approved[0]['human_review']['reviewer']=='Test reviewer'
    summary=json.loads((out/'summary.json').read_text())
    assert (summary['approved'],summary['bad'],summary['unsure'],summary['pending'])==(1,1,1,1)
    assert not summary['training_started'] and summary['dpo_pairs_created']==0

def test_blank_reviews_do_not_turn_ai_drafts_into_human_labels(tmp_path):
    root,file,rows=setup(tmp_path);out=tmp_path/'export'
    review.import_reviews(root/'manifest.json',[file],'Test reviewer',out)
    assert not (out/'approved.jsonl').read_text()
    assert json.loads((out/'summary.json').read_text())['pending']==4

@pytest.mark.parametrize('change',['context','duplicate','source'])
def test_review_identity_and_original_inputs_cannot_silently_change(tmp_path,change):
    root,file,rows=setup(tmp_path)
    rows[0]['decision']='good'
    if change=='context':rows[0]['conversation']='peer: changed message'
    review.write_csv(file,rows)
    if change=='source':
        source=tmp_path/'source.jsonl';source.write_text(source.read_text()+'\n')
    files=[file,file] if change=='duplicate' else [file]
    with pytest.raises(ValueError):review.import_reviews(root/'manifest.json',files,'Test reviewer',tmp_path/'export')
    assert not (tmp_path/'export').exists()
