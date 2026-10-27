import copy
import importlib
import importlib.util
import json

import pytest

from test_reranker_training import inputs
from replycoach.reranker_training import build_training_bundle
from replycoach.schema import Candidate
from replycoach.shared_ranker import PairScore


def api():
    assert importlib.util.find_spec('replycoach.reranker_review'), 'Hard comparison review is missing'
    return importlib.import_module('replycoach.reranker_review')


class Retriever:
    def __init__(self, bundle): self.bundle = bundle
    def retrieve(self, query, top_k):
        return [Candidate([], '', p['reply'], p['style'], 'V0', 1., p['item_id'])
                for p in self.bundle['train'] if p['label']]


class Scorer:
    identity = {'model': 'synthetic-test-scorer'}
    def __init__(self): self.seen = []
    def score_pairs(self, pairs):
        self.seen += pairs
        return [PairScore(float(reply.split()[-1])) for query, reply in pairs]


def make_batch(inputs):
    b = build_training_bundle(inputs)
    s = Scorer()
    batch = api().mine_review_batch(b, Retriever(b), s, count=4)
    return b, batch, s


def test_mining_is_train_only_balanced_and_does_not_assign_labels(inputs):
    b, batch, scorer = make_batch(inputs)
    rows = batch['comparisons']
    assert len(rows) == 4 and len({r['item_id'] for r in rows}) == 4
    assert sorted(r['style'] for r in rows) == ['LOCAL','LOCAL','STANDARD','STANDARD']
    assert all(r['decision']=='pending' and not r['reviewer'] for r in rows)
    assert all(r['item_id'] != r['donor_id'] for r in rows)
    assert not any('incoming 6' in q or 'incoming 7' in q for q, reply in scorer.seen)
    assert all(r['alternative_reply'] != r['reference_reply'] for r in rows)
    other = copy.deepcopy(b)
    other['train'].reverse()
    assert batch == api().mine_review_batch(other, Retriever(other), Scorer(), count=4)


def test_pending_review_cannot_be_exported(inputs):
    b, batch, _ = make_batch(inputs)
    with pytest.raises(ValueError, match='review'):
        api().export_reviewed_pairs(batch,b)


def test_save_is_stale_safe_and_export_strips_names_and_comments(inputs,tmp_path):
    m=api()
    b,batch,_=make_batch(inputs)
    path=tmp_path/'review.json'
    m.create_review_file(path,batch,b)
    first=copy.deepcopy(batch['comparisons'][0])
    with pytest.raises(ValueError,match='reference'):
        m.save_decision(path,b,first,'Human Test Reviewer','unsuitable',False,'PRIVATE_COMMENT')
    m.save_decision(path,b,first,'Human Test Reviewer','suitable',True,'PRIVATE_COMMENT')
    with pytest.raises(ValueError,match='changed'):
        m.save_decision(path,b,first,'Human Test Reviewer','unsuitable',True,'')
    for row in batch['comparisons'][1:]:
        m.save_decision(path,b,row,'Human Test Reviewer','uncertain',False,'')
    final=m.load_review_file(path,b)
    payload=m.export_reviewed_pairs(final,b)
    assert len(payload['pairs'])==1 and payload['pairs'][0]['label']==1
    assert payload['skipped_count']==3
    assert 'Human Test Reviewer' not in json.dumps(payload) and 'PRIVATE_COMMENT' not in json.dumps(payload)
    before=path.read_bytes()
    with pytest.raises(FileExistsError):m.create_review_file(path,batch,b)
    assert path.read_bytes()==before


def test_immutable_tampering_and_foreign_data_are_rejected(inputs,tmp_path):
    m=api()
    b,batch,_=make_batch(inputs)
    path=tmp_path/'review.json'
    m.create_review_file(path,batch,b)
    altered=json.loads(path.read_text(encoding='utf-8'))
    altered['comparisons'][0]['query']='changed'
    path.write_text(json.dumps(altered),encoding='utf-8')
    with pytest.raises(ValueError):m.load_review_file(path,b)
    with pytest.raises(ValueError,match='even|count'):
        m.mine_review_batch(b,Retriever(b),Scorer(),count=3)


def test_foreign_donor_is_rejected_before_any_scoring(inputs):
    b=build_training_bundle(inputs)
    class Foreign:
        def retrieve(self,query,top_k):
            return [Candidate([],'','standard 6','STANDARD','V0',1.,b['manifest']['validation_ids'][0])]
    scorer=Scorer()
    with pytest.raises(ValueError,match='training'):
        api().mine_review_batch(b,Foreign(),scorer,count=4)
    assert scorer.seen==[]
