import importlib
import importlib.util

from streamlit.testing.v1 import AppTest

from test_reranker_training import inputs
from test_reranker_review import make_batch


def review_view():
    from replycoach.reranker_review_app import render
    render()


def setup_view(inputs,tmp_path,monkeypatch):
    assert importlib.util.find_spec('replycoach.reranker_review_app'), 'Reviewer mode is missing'
    ui=importlib.import_module('replycoach.reranker_review_app')
    from replycoach.reranker_review import create_review_file
    b,batch,_=make_batch(inputs)
    path=tmp_path/'review.json'
    create_review_file(path,batch,b)
    monkeypatch.setattr(ui,'DEFAULT_REVIEW_PATH',path)
    monkeypatch.setattr(ui,'load_current_bundle',lambda:b)
    return AppTest.from_function(review_view).run(),b,batch,path


def test_review_requires_a_name_and_saves_only_the_explicit_decision(inputs,tmp_path,monkeypatch):
    app,b,batch,path=setup_view(inputs,tmp_path,monkeypatch)
    assert not app.exception
    save=next(button for button in app.button if button.label=='Save and next')
    assert save.disabled
    app.text_input(key='hard_reviewer_name').set_value('Synthetic Human Reviewer').run()
    row=batch['comparisons'][0]
    key=row['id']
    app.checkbox(key='reference-'+key).check()
    app.selectbox(key='decision-'+key).select('Suitable as another reply')
    next(button for button in app.button if button.label=='Save and next').click().run()
    assert not app.exception
    from replycoach.reranker_review import load_review_file
    saved=load_review_file(path,b)
    assert saved['comparisons'][0]['decision']=='suitable'
    assert all(r['decision']=='pending' for r in saved['comparisons'][1:])
    assert next(button for button in app.button if button.label=='Prepare reviewed training file').disabled


def test_reviewer_cannot_assign_a_label_without_confirming_the_reference(inputs,tmp_path,monkeypatch):
    app,b,batch,path=setup_view(inputs,tmp_path,monkeypatch)
    app.text_input(key='hard_reviewer_name').set_value('Synthetic Human Reviewer').run()
    app.selectbox(key='decision-'+batch['comparisons'][0]['id']).select('Unsuitable reply')
    next(button for button in app.button if button.label=='Save and next').click().run()
    assert not app.exception and app.error
    from replycoach.reranker_review import load_review_file
    assert all(r['decision']=='pending' for r in load_review_file(path,b)['comparisons'])
