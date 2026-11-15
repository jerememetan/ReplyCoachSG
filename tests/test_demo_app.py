from replycoach.demo_app import build_preview_html, parse_preview_line
from html import unescape

import pytest
from streamlit.testing.v1 import AppTest

from replycoach import demo_resources, pipeline, systems
from replycoach.generate import providers
from replycoach.io import ROOT, load_items, write_jsonl


@pytest.fixture
def demo_test(monkeypatch, fake_encoder, tmp_path, request):
    # Missing-provider tests must not depend on a teammate's installed weights.
    monkeypatch.setenv("REPLYCOACH_V4_CHECKPOINT", str(tmp_path / "missing_v4.pt"))
    monkeypatch.setenv("V5_CHECKPOINT", str(tmp_path / "missing_v5"))
    cached_minilm = demo_resources.get_minilm
    cached_generator = getattr(demo_resources, "get_generator", None)
    demo_resources.get_bm25.clear()
    cached_minilm.clear()
    monkeypatch.setattr(demo_resources, "get_minilm_model", lambda: fake_encoder)
    # Keep AppTest suggestions from appending to the team's experiment log.
    monkeypatch.setattr(pipeline, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(systems, "log_event", lambda *args, **kwargs: None)
    if cached_generator:
        cached_generator.clear()
    entry_point = getattr(request, "param", "demo_experiments.py")
    app = AppTest.from_file(str(ROOT / "src/replycoach" / entry_point), default_timeout=20).run()
    yield app
    demo_resources.get_bm25.clear()
    cached_minilm.clear()
    if cached_generator:
        cached_generator.clear()


def compare_in_app(app):
    app.button(key="compare").click().run()
    assert not app.exception
    return app


def suggestion_keys(app):
    return {button.key for button in app.button if button.key and button.key.startswith("select_")}


def test_parse_preview_line_recognises_speakers_case_insensitively():
    assert parse_preview_line(" You: I can go after class ") == ("you", "I can go after class")
    assert parse_preview_line("peer: see you there") == ("peer", "see you there")


def test_parse_preview_line_keeps_unlabelled_message_neutral():
    assert parse_preview_line("lecture ending soon sia") == ("neutral", "lecture ending soon sia")


def test_preview_html_wraps_all_messages_in_one_chat_container():
    html = build_preview_html("Peer: First\nYou: Second", "Third")
    assert html.startswith('<div class="chat-preview">')
    assert '<div class="message-row peer">' in html
    assert '<div class="message-row you">' in html
    assert '<div class="message-row peer"><div class="bubble incoming">Third</div></div>' in html
    assert html.endswith("</div>")


def test_preview_html_shows_empty_display_only_composer():
    html = build_preview_html("Peer: First", "Second")
    assert 'class="preview-composer empty"' in html
    assert "Message to send…" in html


def test_preview_html_shows_selected_reply_in_composer():
    html = build_preview_html("Peer: First", "Second", selected_reply="Sure, see you there.")
    assert 'class="preview-composer selected"' in html
    assert "Sure, see you there." in html


def test_demo_defaults_to_reviewed_training_bank_without_loading_model(demo_test, fake_encoder):
    assert not demo_test.exception
    captions = " ".join(caption.value for caption in demo_test.caption)
    assert demo_test.selectbox(key='bank_choice').value == 'Reviewed development file'
    assert demo_test.text_input(key='development_path').value == 'data/dev/train.jsonl'
    assert 'Reviewed development examples' in captions
    assert '190 examples' in captions
    assert not demo_test.error
    assert not fake_encoder.calls
    assert demo_test.button(key="compare").label == "Compare reply suggestions"


def test_pilot_bank_remains_available_and_switching_clears_suggestions(demo_test):
    app = compare_in_app(demo_test)
    app.selectbox(key='bank_choice').select('Pilot examples (drafts)').run()
    captions = ' '.join(caption.value for caption in app.caption)
    assert 'Pilot examples (drafts)' in captions
    assert '20 examples' in captions
    assert not suggestion_keys(app)
    assert app.session_state.last_comparison is None


def test_comparison_introduces_the_reply_selection_section(demo_test):
    app = compare_in_app(demo_test)
    assert "Suggested replies" in [heading.value for heading in app.header]
    assert "Click a reply to fill the conversation preview." in [
        caption.value for caption in app.caption
    ]


def test_reply_options_use_secondary_buttons_but_compare_stays_primary(demo_test):
    app = compare_in_app(demo_test)
    assert app.button(key="compare").proto.type == "primary"
    assert len(suggestion_keys(app)) == 4
    for key in suggestion_keys(app):
        assert app.button(key=key).proto.type == "secondary"


@pytest.mark.parametrize("method,style", [("bm25", "standard"), ("bm25", "local"), ("minilm", "standard"), ("minilm", "local")])
def test_each_method_style_button_fills_display_only_composer(demo_test, method, style):
    app = compare_in_app(demo_test)
    assert suggestion_keys(app) == {"select_bm25_standard", "select_bm25_local", "select_minilm_standard", "select_minilm_local"}
    assert [heading.value for heading in app.subheader][:2] == ["BM25", "MiniLM"]
    assert "not directly comparable" in " ".join(caption.value for caption in app.caption)
    assert all("V6" not in caption.value for caption in app.caption)
    before = app.text_area(key="conversation").value
    selected = app.button(key=f"select_{method}_{style}").label
    app.button(key=f"select_{method}_{style}").click().run()
    assert app.session_state.selected_reply == selected
    assert app.text_area(key="conversation").value == before
    assert any('preview-composer selected' in block.value and selected in unescape(block.value) for block in app.markdown)
    assert len(app.dataframe) == 2


def test_demo_minilm_failure_is_visible_and_retry_recovers(demo_test, monkeypatch):
    actual_minilm = demo_resources.get_minilm

    def unavailable(bank_json):
        raise RuntimeError("model temporarily unavailable")

    monkeypatch.setattr(demo_resources, "get_minilm", unavailable)
    app = compare_in_app(demo_test)
    assert suggestion_keys(app) == {"select_bm25_standard", "select_bm25_local"}
    assert any("MiniLM unavailable: model temporarily unavailable" in warning.value for warning in app.warning)
    monkeypatch.setattr(demo_resources, "get_minilm", actual_minilm)
    app.button(key="retry_comparison").click().run()
    assert not app.exception
    assert "select_minilm_standard" in suggestion_keys(app)
    assert not app.warning


def test_blank_bank_path_clears_old_suggestions_and_blocks_comparison(demo_test):
    app = compare_in_app(demo_test)
    app.button(key="select_bm25_standard").click().run()
    app.text_input(key="development_path").input("").run()
    assert not suggestion_keys(app)
    assert app.session_state.selected_reply == ""
    assert app.button(key="compare").disabled
    assert app.error


def test_test_bank_cannot_be_selected_for_demo(demo_test, tmp_path):
    row = load_items()[0].to_dict()
    row.update(split="test", status="completed", author="Author", reviewer="Reviewer")
    path = tmp_path / "held_out.jsonl"
    write_jsonl(path, [row])
    app = demo_test
    app.selectbox(key="bank_choice").select("Reviewed development file").run()
    app.text_input(key="development_path").input(str(path)).run()
    assert any("split must be dev" in error.value for error in app.error)
    assert app.button(key="compare").disabled
    assert not suggestion_keys(app)


def test_development_bank_content_edits_clear_stale_comparison(demo_test, tmp_path):
    row = load_items()[0].to_dict()
    row.update(split="dev", status="completed", author="Author", reviewer="Reviewer")
    path = tmp_path / "dev.jsonl"
    write_jsonl(path, [row])
    app = demo_test
    app.selectbox(key="bank_choice").select("Reviewed development file").run()
    app.text_input(key="development_path").input(str(path)).run()
    compare_in_app(app)
    assert app.button(key="select_bm25_standard").label == row["replies"]["standard"]
    row["replies"]["standard"] = "Yes, let's have lunch before the library."
    write_jsonl(path, [row])
    app.run()
    assert not suggestion_keys(app)
    compare_in_app(app)
    assert app.button(key="select_bm25_standard").label == row["replies"]["standard"]


def test_input_edits_clear_stale_suggestions(demo_test):
    app = compare_in_app(demo_test)
    app.text_input(key="incoming").input("can send the notes?").run()
    assert not suggestion_keys(app)
    assert app.session_state.selected_reply == ""


def test_blank_message_disables_comparison(demo_test):
    demo_test.text_input(key="incoming").input("  ").run()
    assert demo_test.button(key="compare").disabled


def test_retrieval_only_mode_has_two_saved_replies_and_no_generation(demo_test, fake_encoder):
    app = demo_test
    app.selectbox(key="system_mode").select("Retrieval only").run()
    compare_in_app(app)
    assert suggestion_keys(app) == {"select_system_standard", "select_system_local"}
    assert app.session_state.last_system_result.effective_mode == "retrieval_only"
    assert app.session_state.last_system_result.generation_count == 0
    assert not fake_encoder.calls


@pytest.mark.parametrize("label,effective", [("Hybrid", "retrieval_only"), ("Generator only", "fallback_only")])
def test_pending_generator_ui_is_honest_and_keeps_fallback_or_retrieval(demo_test, label, effective):
    app = demo_test
    app.selectbox(key="system_mode").select(label).run()
    compare_in_app(app)
    output = app.session_state.last_system_result
    assert output.effective_mode == effective
    assert output.generation_count == 0
    assert any("Role C" in warning.value for warning in app.warning)
    assert any("incomplete" in warning.value.lower() for warning in app.warning)
    assert "select_system_standard" in suggestion_keys(app)


def test_selected_v5_pending_message_names_role_d(demo_test, monkeypatch, tmp_path):
    monkeypatch.setenv("V5_CHECKPOINT", str(tmp_path / "missing"))
    app = demo_test
    app.selectbox(key="system_mode").select("Generator only").run()
    app.selectbox(key="generator_provider").select("V5 (mT5/LoRA)").run()
    compare_in_app(app)
    assert any("Role D" in warning.value for warning in app.warning)


@pytest.mark.parametrize("style", ["standard", "local"])
def test_generated_style_button_still_only_fills_preview(demo_test, monkeypatch, stub_generator, style):
    monkeypatch.setattr(providers, "load_v4", lambda: stub_generator)
    app = demo_test
    app.selectbox(key="system_mode").select("Generator only").run()
    compare_in_app(app)
    assert not app.warning
    assert app.session_state.last_system_result.effective_mode == "generator_only"
    before = app.text_area(key="conversation").value
    text = app.button(key=f"select_system_{style}").label
    app.button(key=f"select_system_{style}").click().run()
    assert app.session_state.selected_reply == text
    assert app.text_area(key="conversation").value == before


def test_mode_changes_clear_existing_comparison_and_selected_reply(demo_test):
    app = compare_in_app(demo_test)
    app.button(key="select_bm25_standard").click().run()
    app.selectbox(key="system_mode").select("Hybrid").run()
    assert not suggestion_keys(app)
    assert app.session_state.selected_reply == ""
    assert app.session_state.last_comparison is None


@pytest.mark.parametrize("change", ["provider", "retriever", "memory", "conversation"])
def test_new_system_settings_and_context_clear_stale_results(demo_test, change):
    app = demo_test
    app.selectbox(key="system_mode").select("Hybrid").run()
    compare_in_app(app)
    app.button(key="select_system_standard").click().run()
    if change == "provider":
        app.selectbox(key="generator_provider").select("V5 (mT5/LoRA)").run()
    elif change == "retriever":
        app.selectbox(key="retrieval_method").select("MiniLM").run()
    elif change == "memory":
        app.selectbox(key="memory_mode").select("M0").run()
    else:
        app.text_area(key="conversation").input("Peer: changed plans\nYou: okay").run()
    assert not suggestion_keys(app)
    assert app.session_state.last_system_result is None
    assert app.session_state.selected_reply == ""


def test_new_system_retry_recovers_after_real_provider_becomes_available(demo_test, monkeypatch, stub_generator):
    app = demo_test
    app.selectbox(key="system_mode").select("Hybrid").run()
    compare_in_app(app)
    assert app.session_state.last_system_result.effective_mode == "retrieval_only"
    monkeypatch.setattr(providers, "load_v4", lambda: stub_generator)
    app.button(key="retry_system").click().run()
    assert not app.exception
    assert app.session_state.last_system_result.effective_mode == "hybrid"
    assert not app.warning


@pytest.mark.parametrize('label,provider,source', [
    ('mT5 A (new handoff)', 'mt5_handoff', 'MT5_A'),
    ('SEA-LION 3B (new handoff)', 'sealion_handoff', 'SEALION'),
])
def test_new_handoff_choices_reach_generator_and_preserve_provenance(
    demo_test, monkeypatch, stub_generator, label, provider, source,
):
    from dataclasses import replace
    calls = []
    stub_generator.source = source
    stub_generator.outputs = [replace(c, source=source) for c in stub_generator.outputs]

    def generator(selected):
        calls.append(selected)
        return stub_generator

    monkeypatch.setattr(demo_resources, 'get_generator', generator)
    app = demo_test
    app.selectbox(key='system_mode').select('Generator only').run()
    app.selectbox(key='generator_provider').select(label).run()
    compare_in_app(app)
    assert not app.exception
    assert calls == [provider]
    result = app.session_state.last_system_result
    assert not result.errors
    assert result.generation_count == 2
    assert result.result.standard.meta['generation_source'] == source


def enable_all_models(app, monkeypatch, stub_generator):
    from replycoach import all_models
    from replycoach.shared_ranker import PairScore

    class Scorer:
        identity = {'model': 'test-only', 'revision': 'fixed'}

        def score_pairs(self, pairs):
            return [PairScore(sum(map(ord, reply)) / 1000) for _, reply in pairs]

    def generator(provider):
        if provider == 'v4':
            return stub_generator
        from test_all_models import Generator
        assert provider == 'sealion_handoff'
        return Generator('SEALION')

    monkeypatch.setattr(demo_resources, 'get_generator', generator)
    monkeypatch.setattr(demo_resources, 'get_shared_ranker', lambda selector='pretrained': Scorer(), raising=False)
    monkeypatch.setattr(all_models, 'log_event', lambda *args, **kwargs: None)
    from replycoach import sealion_comparison
    monkeypatch.setattr(sealion_comparison, 'log_event', lambda *args, **kwargs: None)
    if app.selectbox:
        app.selectbox(key='system_mode').select('All models (experimental)').run()
    return app


@pytest.mark.parametrize('demo_test', ['demo_app.py'], indirect=True)
def test_final_demo_uses_fixed_pipeline_and_collapses_details(demo_test, monkeypatch, stub_generator):
    assert not demo_test.exception
    assert not demo_test.selectbox
    assert demo_test.button(key='compare').label == 'Suggest two replies'
    select_memory = pipeline.select_memory
    modes = []

    def record_mode(session, message, *, mode):
        modes.append(mode)
        return select_memory(session, message, mode=mode)

    monkeypatch.setattr(pipeline, 'select_memory', record_mode)
    app = compare_in_app(enable_all_models(demo_test, monkeypatch, stub_generator))
    output = app.session_state.last_all_models_result
    assert modes == ['M2']
    assert set(output.proposals) == {'BM25', 'MiniLM', 'V4', 'V5'}
    assert suggestion_keys(app) == {'select_all_standard', 'select_all_local'}
    details, = app.expander
    assert details.label == 'Details' and not details.proto.expanded
    assert len(details.dataframe[0].value) == 8


def test_all_models_shows_eight_proposals_and_click_only_fills_composer(demo_test, monkeypatch, stub_generator):
    app = enable_all_models(demo_test, monkeypatch, stub_generator)
    assert app.button(key='compare').label == 'Suggest two replies'
    compare_in_app(app)
    assert not app.exception
    assert any('4/4 models compared' in caption.value for caption in app.caption)
    assert suggestion_keys(app) == {'select_all_standard', 'select_all_local'}
    assert len(app.dataframe[0].value) == 8
    proposals = app.session_state.last_all_models_result.proposals
    assert len(proposals['V5']) == 2
    assert all(c.meta['generation_source'] == 'SEALION' for c in proposals['V5'])
    assert any('V5: SEA-LION 3B' in caption.value for caption in app.caption)
    before = app.text_area(key='conversation').value
    chosen = app.button(key='select_all_standard').label
    app.button(key='select_all_standard').click().run()
    assert app.session_state.selected_reply == chosen
    assert app.text_area(key='conversation').value == before


def test_all_models_input_changes_clear_saved_selection(demo_test, monkeypatch, stub_generator):
    app = enable_all_models(demo_test, monkeypatch, stub_generator)
    compare_in_app(app)
    app.button(key='select_all_local').click().run()
    app.text_input(key='incoming').input('New incoming message').run()
    assert not suggestion_keys(app)
    assert app.session_state.selected_reply == ''
    assert app.session_state.last_all_models_result is None


def test_all_models_partial_source_failure_is_visible(demo_test, monkeypatch, stub_generator):
    app = enable_all_models(demo_test, monkeypatch, stub_generator)

    def generator(provider):
        if provider == 'sealion_handoff':
            raise RuntimeError('V5 checkpoint unavailable')
        return stub_generator

    monkeypatch.setattr(demo_resources, 'get_generator', generator)
    compare_in_app(app)
    assert not app.exception
    assert any('3/4 models available' in warning.value for warning in app.warning)
    assert any('V5 checkpoint unavailable' in warning.value for warning in app.warning)
    assert len(app.dataframe[0].value) == 6
    assert 'V5' not in set(app.dataframe[0].value['Model'])


def test_all_models_skips_ranker_without_editable_phrases_and_has_no_selector(demo_test, monkeypatch, stub_generator):
    app = enable_all_models(demo_test, monkeypatch, stub_generator)
    def forbidden(*args, **kwargs):
        raise AssertionError('The final demo must not load a reranker')
    monkeypatch.setattr(demo_resources, 'get_shared_ranker', forbidden)
    compare_in_app(app)
    assert not app.error
    assert suggestion_keys(app) == {'select_all_standard', 'select_all_local'}
    assert app.button(key='select_all_standard').label == 'SEALION standard'
    assert app.button(key='select_all_local').label == 'SEALION local'
    assert 'shared_selector_checkpoint_v2' not in {s.key for s in app.selectbox}
    assert not any(e.label in {'Shared ranking details','V7 local candidates'} for e in app.expander)
    rows = app.dataframe[0].value
    assert rows.loc[rows['Base reply'],'Model'].tolist() == ['V5','V5']


def test_legacy_selector_state_cannot_override_direct_sealion(demo_test, monkeypatch, stub_generator):
    demo_test.session_state['shared_selector_checkpoint_v2'] = 'V6 (Qwen + DailyDialog)'
    app = enable_all_models(demo_test, monkeypatch, stub_generator)
    compare_in_app(app)
    assert app.button(key='select_all_standard').label == 'SEALION standard'
    assert app.session_state.last_all_models_result.to_dict()['v7']['status']=='skipped'


def test_direct_reply_click_does_not_repeat_generation(demo_test, monkeypatch, stub_generator):
    app = enable_all_models(demo_test, monkeypatch, stub_generator)
    from test_all_models import Generator
    sea = Generator('SEALION')
    monkeypatch.setattr(demo_resources,'get_generator',lambda provider: sea if provider=='sealion_handoff' else stub_generator)
    compare_in_app(app)
    assert len(sea.calls)==1
    before=app.text_area(key='conversation').value
    app.button(key='select_all_local').click().run()
    assert app.session_state.selected_reply=='SEALION local'
    assert app.text_area(key='conversation').value==before
    assert len(sea.calls)==1


def test_sealion_failure_preserves_comparison_and_retry_restores_final_replies(demo_test, monkeypatch, stub_generator):
    app=enable_all_models(demo_test,monkeypatch,stub_generator)
    good=demo_resources.get_generator
    def unavailable(provider):
        if provider=='sealion_handoff':raise RuntimeError('SEA-LION checkpoint unavailable')
        return good(provider)
    monkeypatch.setattr(demo_resources,'get_generator',unavailable)
    compare_in_app(app)
    assert not suggestion_keys(app)
    assert len(app.dataframe[0].value)==6
    assert any('SEA-LION final replies unavailable' in e.value for e in app.error)
    monkeypatch.setattr(demo_resources,'get_generator',good)
    app.button(key='retry_all_models').click().run()
    assert not app.exception and not app.error
    assert app.button(key='select_all_standard').label=='SEALION standard'


@pytest.mark.parametrize(('local_reply', 'candidate_count'), [
    ('Eat!', 2),
    ('Please!', 3),
    ('Never mind, please meet tomorrow.', 7),
])
def test_local_wording_refinement_only_scores_sealion_variants(
    demo_test, monkeypatch, stub_generator, local_reply, candidate_count,
):
    from replycoach.schema import Candidate
    from replycoach.shared_ranker import PairScore
    app=enable_all_models(demo_test,monkeypatch,stub_generator)
    class Sea:
        source='SEALION'
        def generate(self,message,context):
            return [Candidate([],message,'Original standard unchanged.','STANDARD',self.source,.5),
                    Candidate([],message,local_reply,'LOCAL',self.source,.5)]
    calls=[]
    class Scorer:
        identity={'model':'test-only','revision':'fixed'}
        def score_pairs(self,pairs):
            calls.append(pairs)
            return [PairScore(10 if text==local_reply else 0) for _,text in pairs]
    monkeypatch.setattr(demo_resources,'get_generator',lambda p:Sea() if p=='sealion_handoff' else stub_generator)
    monkeypatch.setattr(demo_resources,'get_shared_ranker',lambda selector:Scorer())
    compare_in_app(app)
    assert app.button(key='select_all_standard').label=='Original standard unchanged.'
    assert app.button(key='select_all_local').label==local_reply
    assert len(calls)==1 and len(calls[0])==candidate_count
    output=app.session_state.last_all_models_result.to_dict()
    assert output['v7']['method']=='lexical_phrase_diversity_v3'
    assert output['v7']['candidate_count']==candidate_count
    assert any(h.value=='Local wording candidates' for h in app.subheader)


def test_local_ranker_error_keeps_sealion_and_retry_recovers(demo_test,monkeypatch,stub_generator):
    from replycoach.schema import Candidate
    from replycoach.shared_ranker import PairScore
    app=enable_all_models(demo_test,monkeypatch,stub_generator)
    class Sea:
        source='SEALION'
        def generate(self,message,context):
            return [Candidate([],message,'Standard intact.','STANDARD',self.source,.5),
                    Candidate([],message,'Never mind, please meet tomorrow.','LOCAL',self.source,.5)]
    monkeypatch.setattr(demo_resources,'get_generator',lambda p:Sea() if p=='sealion_handoff' else stub_generator)
    def failed(selector):raise RuntimeError('Local Qwen unavailable')
    monkeypatch.setattr(demo_resources,'get_shared_ranker',failed)
    compare_in_app(app)
    assert app.button(key='select_all_standard').label=='Standard intact.'
    assert app.button(key='select_all_local').label=='Never mind, please meet tomorrow.'
    assert any('Local Qwen unavailable' in w.value for w in app.warning)
    class Scorer:
        identity={'model':'test-only','revision':'fixed'}
        def score_pairs(self,pairs):return [PairScore(1) for _ in pairs]
    monkeypatch.setattr(demo_resources,'get_shared_ranker',lambda selector:Scorer())
    app.button(key='retry_all_models').click().run()
    assert not app.exception and not app.warning
    assert app.session_state.last_all_models_result.local_refinement.complete
