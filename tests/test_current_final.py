import pytest
from replycoach.current_final import make_current_packet
from replycoach.final_test import validate_test_rows


def fixture():
    row=dict(id='test-1',split='test',status='completed',author='A',reviewer='B',privacy_ok=True,
             session_memory=[],message={'text':'fixture question'},replies={'standard':'fixture','local':'fixture'})
    item=validate_test_rows([row],{'test-1'},[])[0]
    def candidate(style,text):return dict(style=style,candidate_text=text,score=.5,meta={})
    std,local=candidate('STANDARD','same standard'),candidate('LOCAL','same local')
    out=dict(complete=True,source_errors={},ranking_error=None,
             proposals={s:[std.copy(),local.copy()] for s in ['BM25','MiniLM','V4','V5']},
             standard=std,local=candidate('LOCAL','same local lah'),
             v7=dict(complete=True,original=local,selected=candidate('LOCAL','same local lah')))
    return item,out


def test_current_packet_covers_six_systems_and_preserves_v6_v7_standard():
    item,out=fixture()
    public,private=make_current_packet([item],{item.id:out})
    assert len(public)==len(private)==12
    assert {r['system'] for r in private}=={'BM25','MiniLM','V4','V5','V6','V7'}
    ids={r['candidate_id']:r for r in public}
    lookup={(r['system'],r['style']):ids[r['candidate_id']]['reply'] for r in private}
    assert lookup['V6','STANDARD']==lookup['V7','STANDARD']=='same standard'
    assert lookup['V6','LOCAL']=='same local'
    assert lookup['V7','LOCAL']=='same local lah'
    assert all('system' not in r and 'candidate' not in r for r in public)
    assert all(r['evaluation_split']=='test' for r in private)


@pytest.mark.parametrize('failure',['partial','v7','missing_style'])
def test_current_packet_refuses_partial_outputs(failure):
    item,out=fixture()
    if failure=='partial':out['complete']=False
    elif failure=='v7':out['v7']['complete']=False
    else:out['proposals']['V5']=out['proposals']['V5'][:1]
    with pytest.raises(ValueError):make_current_packet([item],{item.id:out})


def test_packet_roundtrips_existing_judge_and_only_reuses_identical_tasks(tmp_path):
    import importlib.util
    from replycoach.io import ROOT
    from replycoach.judging import write_judging_packet
    from replycoach.transformer_comparison import task_key
    spec=importlib.util.spec_from_file_location('current_packet_judge',ROOT/'scripts/judge_replies.py')
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    item,out=fixture()
    public,private=make_current_packet([item],{item.id:out})
    packet=tmp_path/'packet';write_judging_packet(packet,public,private)
    loaded,mapping,_=judge.load_packet(packet)
    assert loaded==public
    selected={(mapping[r['candidate_id']]['system'],r['style']):r for r in loaded}
    assert task_key(selected['V6','STANDARD'],'p1')==task_key(selected['V7','STANDARD'],'p1')
    assert task_key(selected['V6','LOCAL'],'p1')!=task_key(selected['V7','LOCAL'],'p1')
    assert task_key(selected['V7','LOCAL'],'p1')!=task_key(selected['V7','LOCAL'],'p2')
