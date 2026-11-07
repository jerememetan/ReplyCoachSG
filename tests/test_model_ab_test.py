import importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('model_ab',Path(__file__).parents[1]/'scripts/model_ab_test.py')
ab=importlib.util.module_from_spec(spec);spec.loader.exec_module(ab)

def test_blind_packet_matches_sources_and_is_stable(tmp_path):
    items=[dict(id=f'review-{i}',session_memory=[dict(speaker='peer',text='Earlier turn')],
                message=dict(text=f'Incoming {i}')) for i in range(3)]
    items_path=tmp_path/'items.jsonl'
    items_path.write_text(''.join(json.dumps(item)+'\n' for item in items))
    sources={}
    for name in ab.SOURCES:
        path=tmp_path/(name+'.json')
        rows=[dict(item_id=item['id'],generated_standard=f'{name} reply to {item["id"]}') for item in items]
        field='development_generations' if name=='default' else 'validation_outputs'
        path.write_text(json.dumps({field:rows}));sources[name]=path
    directory=tmp_path/'review'
    packet=ab.prepare(directory,sources=sources,items_path=items_path)
    key=json.loads((directory/'key.json').read_text())
    assert len(packet['questions'])==len(items)*(len(sources)-1)
    for q in packet['questions']:
        assert set(key['mapping'][q['id']].values())=={'default',q['comparison']}
        for letter in ('A','B'):
            name=key['mapping'][q['id']][letter]
            report=json.loads(sources[name].read_text())
            rows=report.get('development_generations',report.get('validation_outputs'))
            assert q[letter]==next(r['generated_standard'] for r in rows if r['item_id']==q['item_id'])
    assert ab.prepare(directory,sources=sources,items_path=items_path)==packet

def test_votes_persist_and_can_be_corrected(tmp_path):
    ab.save_vote(tmp_path,'case1','A','supported')
    ab.save_vote(tmp_path,'case2','Both bad','wrong role')
    ab.save_vote(tmp_path,'case1','Tie','both valid')
    rows=json.loads((tmp_path/'ratings.json').read_text())
    assert rows=={'case1':{'choice':'Tie','notes':'both valid'},'case2':{'choice':'Both bad','notes':'wrong role'}}
    with pytest.raises(ValueError):ab.save_vote(tmp_path,'case3',None,'')
