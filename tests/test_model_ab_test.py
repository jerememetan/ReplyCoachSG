import importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('model_ab',Path(__file__).parents[1]/'scripts/model_ab_test.py')
ab=importlib.util.module_from_spec(spec);spec.loader.exec_module(ab)

def test_blind_packet_matches_sources_and_is_stable(tmp_path):
    packet=ab.prepare(tmp_path)
    key=json.loads((tmp_path/'key.json').read_text())
    assert len(packet['questions'])==150
    for q in packet['questions']:
        assert set(key['mapping'][q['id']].values())=={'default',q['comparison']}
        for letter in ('A','B'):
            name=key['mapping'][q['id']][letter]
            report=json.loads(ab.SOURCES[name].read_text())
            rows=report.get('development_generations',report.get('validation_outputs'))
            assert q[letter]==next(r['generated_standard'] for r in rows if r['item_id']==q['item_id'])
    assert ab.prepare(tmp_path)==packet

def test_votes_persist_and_can_be_corrected(tmp_path):
    ab.save_vote(tmp_path,'case1','A','supported')
    ab.save_vote(tmp_path,'case2','Both bad','wrong role')
    ab.save_vote(tmp_path,'case1','Tie','both valid')
    rows=json.loads((tmp_path/'ratings.json').read_text())
    assert rows=={'case1':{'choice':'Tie','notes':'both valid'},'case2':{'choice':'Both bad','notes':'wrong role'}}
    with pytest.raises(ValueError):ab.save_vote(tmp_path,'case3',None,'')
