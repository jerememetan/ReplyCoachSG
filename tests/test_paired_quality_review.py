import json,sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_paired_quality_review import prepare
from score_paired_quality_review import score

def packet(tmp_path):
    items=[dict(id=f'I{i:02}',session_memory=[],message={'text':f'Question {i}'}) for i in range(50)]
    validation=tmp_path/'validation.jsonl';validation.write_text(''.join(json.dumps(c)+'\n' for c in items))
    for arm in ('old','new'):
        rows=[dict(item_id=c['id'],generated_standard=arm+' standard',generated_local=arm+' local') for c in items]
        (tmp_path/f'{arm}_s425.json').write_text(json.dumps(dict(optimizer_steps=784,final_validation_outputs=rows)))
    out=tmp_path/'review';prepare(tmp_path,['old','new'],out,validation)
    return out

def test_whole_pairs_are_kept_together_and_references_hidden(tmp_path):
    p=packet(tmp_path);data=json.loads((p/'packet.json').read_text())
    assert len(data['questions'])==50
    for q in data['questions']:
        for pair in q['options'].values():
            assert pair['standard'].split()[0]==pair['local'].split()[0]
            assert set(pair)=={'standard','local'}

def test_missing_axis_or_missing_pair_cannot_be_scored(tmp_path):
    from hashlib import sha256
    p=packet(tmp_path);public=p/'packet.json';r=p/'ratings.json'
    r.write_text(json.dumps(dict(packet_sha256=sha256(public.read_bytes()).hexdigest(),ratings=[])))
    with pytest.raises(ValueError,match='Every whole pair'):score(public,p/'key.json',r)

def test_same_meaning_cannot_compensate_for_failed_singlish(tmp_path):
    from hashlib import sha256
    p=packet(tmp_path);public=p/'packet.json';data=json.loads(public.read_text());r=p/'ratings.json'
    rows=[dict(n=q['n'],option=letter,context='pass',same_meaning='pass',singlish='fail',reason='Neutral English.')
          for q in data['questions'] for letter in q['options']]
    r.write_text(json.dumps(dict(packet_sha256=sha256(public.read_bytes()).hexdigest(),ratings=rows)))
    result=score(public,p/'key.json',r)
    assert all(c['all_three_pass']==0 and c['same_meaning_pass']==50 for c in result['counts'].values())
