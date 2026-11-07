import hashlib,importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('duration_review',Path(__file__).resolve().parents[1]/'scripts/review_duration_probe_v1.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)

def test_all_eight_checkpoints_required_and_all_axes_scored(tmp_path):
    source=tmp_path/'cases.jsonl'
    items=[dict(id=f'Q{i}',session_memory=[],message={'text':f'Question {i}'}) for i in range(24)]
    source.write_text(''.join(json.dumps(c)+'\n' for c in items))
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    for arm in ['control','capacity']:
        for step in review.STEPS:
            (tmp_path/f'{arm}_step{step}.json').write_text(json.dumps(dict(arm=arm,optimizer_step=step,seed=425,fresh_source_sha256=digest,outputs=[dict(item_id=c['id'],generated_standard='Okay.',generated_local='Can lah.') for c in items])))
    out=tmp_path/'review';review.prepare(tmp_path,source,out)
    packet=json.loads((out/'packet.json').read_text());assert all(len(q['options'])==8 for q in packet['questions'])
    ratings=dict(packet_sha256=hashlib.sha256((out/'packet.json').read_bytes()).hexdigest(),ratings=[dict(n=q['n'],option=o,standard='pass',local='pass',same_meaning='fail',singlish='pass',reason='Changed meaning.') for q in packet['questions'] for o in q['options']])
    (out/'ratings.json').write_text(json.dumps(ratings));review.score(out)
    result=json.loads((out/'summary.json').read_text());assert len(result['counts'])==8
    assert all(c['pairs']==24 and c['all_pass']==0 for c in result['counts'].values())
    ratings['ratings'].pop();(out/'ratings.json').write_text(json.dumps(ratings))
    with pytest.raises(AssertionError):review.score(out)
