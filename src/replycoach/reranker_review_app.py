"""Reranker comparison view embedded in the existing team reviewer."""
import copy
from html import escape
import json
from pathlib import Path

import streamlit as st

from replycoach.evaluation import DEFAULT_BANK, DEFAULT_QUERIES, load_evaluation_inputs
from replycoach.io import ROOT
from replycoach.reranker_review import export_reviewed_pairs, load_review_file, save_decision
from replycoach.reranker_training import build_training_bundle

DEFAULT_REVIEW_PATH = ROOT/'artifacts/reranker_v6/hard_review/review.json'
LABELS = {'Choose a decision': 'pending', 'Suitable as another reply': 'suitable',
          'Unsuitable reply': 'unsuitable', 'Uncertain — skip': 'uncertain',
          'Reference needs correction — skip': 'reference_needs_correction'}


def load_current_bundle():
    return build_training_bundle(load_evaluation_inputs(DEFAULT_QUERIES, DEFAULT_BANK))


def render():
    st.title('Reranker reply comparisons')
    st.caption('Review difficult alternatives from training conversations. More than one reply can be suitable.')
    reviewer = st.text_input('Your name', key='hard_reviewer_name', placeholder='Enter your name to save decisions')
    with st.expander('Review file'):
        selected = st.text_input('Comparison file', str(DEFAULT_REVIEW_PATH), key='hard_review_file')
    path = Path(selected)
    try:
        bundle = load_current_bundle()
        batch = load_review_file(path, bundle)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f'Cannot load comparisons: {exc}')
        st.caption('Create a queue with scripts/prepare_reranker_review.py before opening this view.')
        return
    rows = batch['comparisons']
    completed = sum(r['decision'] != 'pending' for r in rows)
    st.progress(completed/len(rows), text=f'{completed} of {len(rows)} comparisons reviewed')
    if 'hard_next_index' in st.session_state:
        st.session_state.hard_review_index = st.session_state.pop('hard_next_index')
    index = st.selectbox('Comparison', range(len(rows)), key='hard_review_index',
                         format_func=lambda i: f"{i+1} / {len(rows)} · {rows[i]['item_id']} · {rows[i]['style']}")
    latest = rows[index]
    identity = str(path.resolve()), batch['identity_sha256'], latest['id']
    if st.session_state.get('hard_opened_identity') != identity:
        st.session_state.hard_opened_identity = identity
        st.session_state.hard_opened_case = copy.deepcopy(latest)
    row = st.session_state.hard_opened_case
    key = row['id']
    if row != latest:
        st.warning('This comparison changed after you opened it. Reload before saving.')
    if st.button('Reload comparison'):
        st.session_state.pop('hard_opened_identity', None)
        for widget in ['reference-', 'decision-', 'notes-']:
            st.session_state.pop(widget+key, None)
        st.rerun()
    if st.session_state.get('hard_review_saved'):
        st.success(st.session_state.pop('hard_review_saved'))
    st.caption(f"{row['style']} · Training item {row['item_id']} · Status: {row['decision']}")
    context = row['query'].removeprefix('Conversation (oldest first):\n').rsplit('\nRequested reply style:', 1)[0]
    st.text_area('Conversation and incoming peer message seen by the reranker', context, disabled=True, height=160)
    left, right = st.columns(2)
    with left:
        st.subheader('Reference reply')
        st.caption('Previously reviewed. Confirm it still fits this context and style.')
        st.markdown(f'<p style="font-size:1.3rem;line-height:1.5">{escape(row["reference_reply"])}</p>', unsafe_allow_html=True)
    with right:
        st.subheader('Alternative to review')
        st.caption('A real reply from another training item. It may also be suitable here.')
        st.markdown(f'<p style="font-size:1.3rem;line-height:1.5">{escape(row["alternative_reply"])}</p>', unsafe_allow_html=True)
    with st.form('hard-comparison-'+key):
        reference_ok = st.checkbox('The reference reply is appropriate for this conversation and style',
                                   value=row['reference_ok'], key='reference-'+key)
        label = next(label for label, value in LABELS.items() if value == row['decision'])
        decision = st.selectbox('Is the alternative suitable here?', list(LABELS),
                                index=list(LABELS).index(label), key='decision-'+key)
        notes = st.text_area('Review notes (optional)', row['notes'], key='notes-'+key, height=80)
        save = st.form_submit_button('Save and next', type='primary', disabled=not reviewer.strip())
    if save:
        try:
            save_decision(path, bundle, row, reviewer, LABELS[decision], reference_ok, notes)
        except (OSError, ValueError) as exc:
            st.error(f'Not saved: {exc}')
        else:
            st.session_state.hard_review_saved = f"Saved {row['item_id']} · {row['style']}"
            st.session_state.pop('hard_opened_identity', None)
            st.session_state.hard_next_index = min(index+1, len(rows)-1)
            st.rerun()
    with st.expander('Why this comparison was selected'):
        st.write('BM25 found a nearby training reply. V6 ranked it highly relative to the reference; this does not supply its label.')
        st.caption(f"Reference logit {row['reference_score']:.3f} · Alternative logit {row['alternative_score']:.3f} · Donor {row['donor_id']}")
    can_export = completed == len(rows) and any(r['decision'] in {'suitable', 'unsuitable'} for r in rows)
    if st.button('Prepare reviewed training file', disabled=not can_export):
        try:
            payload = export_reviewed_pairs(batch, bundle)
            output = path.parent/f"reviewed_pairs_{payload['source_review_sha256'][:12]}.json"
            text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            if output.exists():
                if output.read_text(encoding='utf-8') != text:
                    raise ValueError('existing export differs; do not overwrite it')
            else:
                with output.open('x', encoding='utf-8') as handle:
                    handle.write(text)
            st.success(f'Reviewed training file ready: {output}')
        except (OSError, ValueError) as exc:
            st.error(f'Not exported: {exc}')
    st.caption('Suitable alternatives become extra positives; unsuitable ones become reviewed negatives. Skips are excluded. All comparisons need a decision before export.')
