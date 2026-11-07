import pytest
from replycoach.generate.sealion_public_v1 import assign_groups,clean,sample,pair_key
from replycoach.generate.sealion_data import source


def raw(context=None,message='Can meet later?',response='tomorrow can lah',chat='a',speakers=None):
    return dict(context=context or [],message=message,response=response,chat=chat,speakers=speakers or [])


def test_dd_whole_dialogues_reconstructed_independent_of_split_names():
    data=[raw(message='Hello there',response='How are you'),raw(['Hello there'],message='How are you',response='I am well'),raw(message='New conversation',response='Good to see you')]
    rows=assign_groups(data,'dailydialog')
    assert rows[0]['group_id']==rows[1]['group_id']!=rows[2]['group_id']
    assert assign_groups(data[:2],'dailydialog')[0]['group_id']==rows[0]['group_id']
    with pytest.raises(ValueError):assign_groups([raw(['Something'],message='Broken order')],'dailydialog')


def test_cosem_visible_roles_filter_preserves_wording_without_corpus_style_label():
    good=assign_groups([raw(['We could meet','I finish at six'],speakers=['peer','user'])],'cosem')[0]
    row,reason=clean(good,'cosem');assert reason is None and row['response']=='tomorrow can lah'
    assert row['style']=='DIALOGUE'
    assert source(row,'DIALOGUE')==source({**row,'corpus':'dailydialog'},'DIALOGUE')
    assert clean({**good,'speakers':['other','user']},'cosem')[1]=='non_two_party_visible_window'


def test_filter_contacts_artifacts_and_filler():
    for response in ['send it to person@example.com','{{name}} can meet','haha','click https://example.org']:
        row=assign_groups([raw(response=response)],'cosem')[0]
        assert clean(row,'cosem')[0] is None


def test_sampling_caps_groups_repeated_responses_and_duplicate_pairs():
    pool=[]
    for group in range(5):
        for i in range(8):
            row=assign_groups([raw(context=['Earlier message','My response'],speakers=['peer','user'],message=f'Question {group} number {i}',response=f'answer number {i}',chat=str(group))],'cosem')[0]
            clean_row,_=clean(row,'cosem');pool.extend([clean_row,dict(clean_row)])
    chosen=sample(pool,12,group_cap=3,response_cap=2)
    assert len({pair_key(r) for r in chosen})==len(chosen)
    assert all(sum(x['group_id']==r['group_id'] for x in chosen)<=3 for r in chosen)
    assert all(sum(x['response']==r['response'] for x in chosen)<=2 for r in chosen)
