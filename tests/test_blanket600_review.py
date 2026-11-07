import hashlib,importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('blanket_review',Path(__file__).resolve().parents[1]/'scripts/review_blanket600_v1.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)

def test_primary_final_outputs_require_full_cases_and_four_axes(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    reports={arm:dict(optimizer_steps=1120,preflight={'common':True}) for arm in ['control','expanded']}
    for split,count,path,field in [('validation',50,'data/dev/validation.jsonl','validation_outputs'),('diagnostic',24,'data/reward_diagnostics/duration_probe_v1.jsonl','diagnostic_outputs')]:
        items=[dict(id=f'{split}{i}',session_memory=[],message={'text':f'Question{i}'}) for i in range(count)]
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(c)+'\n' for c in items))
        for arm in reports:reports[arm][field]=[dict(item_id=c['id'],generated_standard='Yes.',generated_local='Can lah.') for c in items]
    for arm,r in reports.items():Path(f'{arm}_final_outputs.json').write_text(json.dumps(r))
    out=Path('review');review.prepare(Path('.'),out)
    p=json.loads((out/'packet.json').read_text());assert len(p['questions'])==74
    ratings=dict(packet_sha256=hashlib.sha256((out/'packet.json').read_bytes()).hexdigest(),ratings=[dict(n=q['n'],option=o,standard='pass',local='fail',same_meaning='pass',singlish='pass',reason='Wrong role.') for q in p['questions'] for o in q['options']])
    (out/'ratings.json').write_text(json.dumps(ratings));review.score(out)
    summary=json.loads((out/'summary.json').read_text());assert all(c['all_pass']==0 for group in summary['counts'].values() for c in group.values())
    ratings['ratings'].pop();(out/'ratings.json').write_text(json.dumps(ratings))
    with pytest.raises(AssertionError):review.score(out)
