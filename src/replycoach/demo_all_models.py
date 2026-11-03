"""Existing demo's experimental all-model selection view."""
import json

import streamlit as st

from replycoach.all_models import AllModelsResult


def render_all_models(output: AllModelsResult, select_reply, *, selector_label: str) -> None:
    st.divider()
    st.header('Selected replies')
    count = len(output.available_sources)
    if output.complete:
        st.caption('4/4 models compared · one shared ranking model · experimental')
    elif count < 4:
        st.warning(f'Partial comparison: {count}/4 models available.')
    else:
        st.caption('4/4 models produced proposals')
    for source, error in output.source_errors.items():
        st.warning(f'{source} unavailable: {error}')
    if output.ranking_error:
        st.error(f'Shared ranking unavailable: {output.ranking_error}')
    refinement = output.local_refinement
    if refinement and refinement.error:
        st.warning(f'V7 local refinement unavailable: {refinement.error}. Showing the original local reply.')
    if not output.proposals:
        st.error('No model proposals are available. Check the model setup and retry.')
        return
    if output.ranking:
        st.caption('Click a reply to fill the conversation preview.')
        for column, label, style, candidate in zip(
            st.columns(2, gap='large'), ['Standard English', 'Light local'], ['standard', 'local'],
            [output.ranking.standard, output.final_local],
        ):
            if candidate is None:
                continue
            with column:
                st.markdown(f'<p class="reply-style-label">{label}</p>', unsafe_allow_html=True)
                st.button(candidate.candidate_text, key=f'select_all_{style}', type='secondary',
                          use_container_width=True, on_click=select_reply, args=(candidate.candidate_text,))
                st.caption('Proposed by ' + ', '.join(candidate.meta['contributing_models']))
                if candidate.meta['fallback_models']:
                    st.caption('Includes a safety fallback from ' + ', '.join(candidate.meta['fallback_models']))
                if style == 'local' and refinement and refinement.complete:
                    st.caption('V7 kept the original reply.' if candidate.meta['v7']['is_original']
                               else 'V7 selected a lexicon variation.')
        st.caption(f'Selected by {selector_label}. Ranking signals are not confidence percentages.')
    st.subheader('Model proposals')
    rows = []
    for model, candidates in output.proposals.items():
        for candidate in candidates:
            chosen = (getattr(output.ranking, candidate.style.lower()) if output.ranking else None)
            selected = bool(chosen and model in chosen.meta['contributing_models']
                            and ' '.join(candidate.candidate_text.split()) == ' '.join(chosen.candidate_text.split()))
            fallback = candidate.meta.get('candidate_source', candidate.source) == 'fallback'
            rows.append({'Model': model, 'Style': candidate.style.title(), 'Reply': candidate.candidate_text,
                         'Origin': 'Safety fallback' if fallback else 'Retrieved' if model in {'BM25','MiniLM'} else 'Generated',
                         'Selected initially': selected})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if output.ranking:
        with st.expander('Shared ranking details'):
            st.caption('Raw cross-encoder scores; source-model scores do not influence this selection.')
            st.dataframe([{'Style': c.style, 'Reply': c.candidate_text,
                           'Models': ', '.join(c.meta['contributing_models']),
                           'Ranking score': round(c.score, 4), 'Context truncated': c.meta['truncated']}
                          for c in output.ranking.candidates], use_container_width=True, hide_index=True)
            if output.ranking.candidates:
                identity = output.ranking.candidates[0].meta['ranker']
                st.caption(f"{identity.get('model')} · revision {identity.get('revision')}")
                if identity.get('adapter_sha256'):
                    st.caption(f"Adapter SHA-256: {identity['adapter_sha256']}")
            st.write('Source time (seconds):', {k: round(v, 2) for k,v in output.source_seconds.items()})
            st.caption(f'Shared ranking: {output.ranking_seconds:.2f} seconds; initial model loading may be included.')
    if refinement:
        with st.expander('V7 local candidates'):
            if refinement.complete:
                st.caption('Original local winner + 3 distinct lexicon variations, scored by the same selected ranker.')
            else:
                st.caption('V7 did not complete. The original local winner is retained.')
            texts = [refinement.original.candidate_text, *refinement.variations]
            scored = {c.meta['v7']['variant_index']: c for c in refinement.ranking.candidates} if refinement.ranking else {}
            st.dataframe([
                {'Candidate': 'Original' if index == 0 else f'Variation {index}', 'Reply': text,
                 'Ranking score': round(scored[index].score, 4) if index in scored else None,
                 'Selected': text == refinement.selected.candidate_text}
                for index, text in enumerate(texts)
            ], use_container_width=True, hide_index=True)
            st.caption(f'Lexicon variations: {refinement.generation_seconds * 1000:.2f} ms; '
                       f'V7 reranking: {refinement.ranking_seconds:.2f} seconds. '
                       'No additional generator or Gemma calls.')
    st.download_button('Download all-model result (JSON)',
                       json.dumps(output.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
                       file_name='replycoach_all_models.json', mime='application/json', key='download_all_models')
