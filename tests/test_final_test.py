"""Invented fixtures exercise guards only, never training or evaluation gold."""
import pytest
from replycoach.final_test import validate_test_rows, make_packet


def row(identifier='test-1', split='test'):
    return dict(id=identifier, split=split, status='completed', author='A', reviewer='B',
                privacy_ok=True, session_memory=[], message={'text':identifier},
                replies={'standard':'fixture', 'local':'fixture'})


def test_test_loader_requires_exact_frozen_ids_and_rejects_overlap():
    assert len(validate_test_rows([row()], {'test-1'}, [])) == 1
    for rows in [[row(split='dev')], [row(), row()], [row('other')]]:
        with pytest.raises(ValueError):
            validate_test_rows(rows, {'test-1'}, [])
    with pytest.raises(ValueError, match='overlap'):
        validate_test_rows([row()], {'test-1'}, [row()])
    unsafe=row();unsafe['privacy_ok']=False
    with pytest.raises(ValueError): validate_test_rows([unsafe], {'test-1'}, [])


def test_packet_has_no_gold_or_model_identity_and_matches_v6_to_same_source_text():
    item=validate_test_rows([row()], {'test-1'}, [])[0]
    candidate={'candidate_text':'fixture answer','style':'STANDARD','score':.2}
    local={**candidate,'style':'LOCAL'}
    output={'proposals':{s:[candidate,local] for s in ['BM25','MiniLM','V4','V5']},
            'standard':candidate,'local':local}
    public,private,selections=make_packet([item], {'test-1':output})
    assert len(public)==len(private)==8
    assert len(selections)==2
    assert {s['candidate_id'] for s in selections} <= {p['candidate_id'] for p in public}
    assert all(set(p)=={'schema_version','candidate_id','conversation_id','conversation',
                         'incoming_message','style','reply'} for p in public)
    assert all(p['evaluation_split']=='test' for p in private)
