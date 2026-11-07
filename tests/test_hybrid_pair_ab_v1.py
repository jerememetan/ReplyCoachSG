import copy,hashlib,importlib.util,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from hybrid_pair_ab_test import prepare_packet,save_vote
from prepare_hybrid_pair_ab_v1 import fresh_cases


def fixture_packet(tmp_path):
    cases=fresh_cases()[:2];folder=tmp_path/'review';folder.mkdir()
    (folder/'inputs.json').write_text(json.dumps(cases))
    outputs=[]
    for c in cases:
        outputs.append(dict(item_id=c['item_id'],hybrid=dict(standard='I can meet later.',local='Later can meet.',local_conditioned_on='I can meet later.'),earlier_paired=dict(standard='I will be there.',local='I go there.',local_conditioned_on='I will be there.')))
    report=tmp_path/'report.json';report.write_text(json.dumps(dict(status='complete',inputs=cases,outputs=outputs)))
    packet=prepare_packet(folder,report);return folder,report,packet


def test_pair_packet_is_complete_blinded_and_stable(tmp_path):
    folder,report,packet=fixture_packet(tmp_path)
    key=json.loads((folder/'key.json').read_text());source=json.loads(report.read_text())
    assert len(packet['questions'])==2
    for q,original in zip(packet['questions'],source['outputs']):
        assert set(q['options'])=={'A','B'}
        for letter,arm in key['mapping'][q['id']].items():
            assert q['options'][letter]=={k:original[arm][k] for k in ('standard','local')}
    assert 'hybrid' not in json.dumps(packet) and 'earlier_paired' not in json.dumps(packet)
    assert prepare_packet(folder,report)==packet


def test_corrupted_packet_and_wrong_conditioning_are_rejected(tmp_path):
    folder,report,packet=fixture_packet(tmp_path)
    (folder/'packet.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):prepare_packet(folder,report)
    (folder/'packet.json').unlink();(folder/'key.json').unlink()
    rows=json.loads(report.read_text());rows['outputs'][0]['hybrid']['local_conditioned_on']='Wrong anchor';report.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='anchor'):prepare_packet(folder,report)


def test_votes_and_corrections_persist_without_inferred_pass_labels(tmp_path):
    folder,report,packet=fixture_packet(tmp_path);qid=packet['questions'][0]['id']
    save_vote(folder,qid,'Both bad',{'A':['LOCAL changes STANDARD meaning'],'B':[]},'I can meet tomorrow.','Tomorrow can meet.','Wrong time')
    vote=json.loads((folder/'ratings.json').read_text())[qid]
    assert vote['corrected_local']=='Tomorrow can meet.' and vote['issues']['B']==[]
    assert vote['unmarked_issues']=='Not rated; absence of a flag is not a pass'
    with pytest.raises(ValueError):save_vote(folder,qid,None,{'A':[],'B':[]},'','','')


def test_ui_has_no_preselected_vote_and_saves_only_in_fixture(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    folder,report,packet=fixture_packet(tmp_path)
    monkeypatch.setenv('REPLYCOACH_PAIR_REVIEW_DIR',str(folder))
    app=AppTest.from_file(str(Path(__file__).parents[1]/'scripts/hybrid_pair_ab_test.py')).run(timeout=20)
    assert not app.exception,list(app.exception)
    assert app.radio[0].value is None
    assert not (folder/'ratings.json').exists()
    app.radio[0].set_value('Tie').run()
    next(b for b in app.button if b.label=='Save and next').click().run()
    assert not app.exception,list(app.exception)
    saved=json.loads((folder/'ratings.json').read_text());assert len(saved)==1
    assert saved[packet['questions'][0]['id']]['choice']=='Tie'
