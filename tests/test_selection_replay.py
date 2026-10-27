import copy
import importlib
import importlib.util

import pytest

from replycoach.schema import Candidate
from replycoach.shared_ranker import PairScore


def api():
    assert importlib.util.find_spec('replycoach.selection_replay'), 'Selection replay has not been implemented'
    return importlib.import_module('replycoach.selection_replay')


def fixture():
    public, mapping, ratings = [], {}, []
    for index, model in enumerate(['BM25','MiniLM','V4','V5']):
        for style in ['STANDARD','LOCAL']:
            cid=f'PRIVATE_{index}_{style}'
            text=f'proposal {index} {style.lower()}'
            c=Candidate(['earlier'], 'incoming', text, style, 'V6', .9-index*.1,
                        meta={'candidate_source': ['V0','V2','V4','V5'][index]})
            public.append({'candidate_id':cid,'style':style,'reply':text,
                           'conversation':[{'speaker':'user','text':'earlier'}],
                           'incoming_message':{'speaker':'peer','text':'incoming'}})
            mapping[cid]={'candidate_id':cid,'item_id':'Q1','system':model+'_M2','style':style,
                          'memory_mode':'M2','candidate':c.to_dict()}
            for p in ['P1','P2']:
                ratings.append({'candidate_id':cid,'persona_id':p,'status':'ok',
                                'rating':{'appropriateness':index+1,'naturalness':5,'style_fit':5,
                                          'would_send':index>1}})
    return public,mapping,ratings,['P1','P2']


class Scorer:
    identity={'model':'test-only','revision':'fixed'}

    def score_pairs(self,pairs):
        assert all('PRIVATE_' not in query and 'appropriateness' not in query for query,_ in pairs)
        return [PairScore(2 if 'proposal 1' in reply else -2) for _,reply in pairs]


def test_replay_selects_without_ratings_and_preserves_paired_baselines():
    m=api()
    result=m.replay_selection(*fixture(),Scorer(),bootstrap_samples=40)
    rows={(r['policy'],r['style']):r for r in result['summary']}
    assert rows['shared','STANDARD']['appropriateness']==2
    assert rows['fixed_v5','STANDARD']['appropriateness']==4
    assert rows['fixed_minilm','LOCAL']['appropriateness']==2
    assert rows['legacy','STANDARD']['appropriateness']==1
    assert len(result['selections'])==2
    assert all(r['shared_models']==['MiniLM'] for r in result['selections'])
    assert result['record_type']=='all_models_selection_replay'
    assert result['comparison']['shared_minus_fixed_v5']['STANDARD']['appropriateness']['difference']==-2


@pytest.mark.parametrize('mutation',['missing_rating','duplicate_rating','missing_candidate','wrong_text'])
def test_replay_rejects_incomplete_or_changed_evidence(mutation):
    m=api()
    public,mapping,ratings,personas=fixture()
    if mutation=='missing_rating': ratings.pop()
    elif mutation=='duplicate_rating': ratings.append(copy.deepcopy(ratings[0]))
    elif mutation=='missing_candidate': public.pop()
    else: public[0]['reply']='changed text'
    with pytest.raises(ValueError):
        m.replay_selection(public,mapping,ratings,personas,Scorer(),bootstrap_samples=40)


def test_replay_scores_do_not_change_when_evaluation_labels_change():
    m=api()
    first=fixture()
    second=copy.deepcopy(first)
    for row in second[2]:
        row['rating']['appropriateness']=5-row['rating']['appropriateness']
        row['rating']['would_send']=not row['rating']['would_send']
    a=m.replay_selection(*first,Scorer(),bootstrap_samples=40)
    b=m.replay_selection(*second,Scorer(),bootstrap_samples=40)
    assert [(r['shared_candidate_id'],r['shared_score']) for r in a['selections']]==[
        (r['shared_candidate_id'],r['shared_score']) for r in b['selections']]
