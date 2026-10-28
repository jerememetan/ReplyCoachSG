from replycoach.demo_app import build_preview_html, parse_preview_line

import pytest
from streamlit.testing.v1 import AppTest

from replycoach import demo_resources, pipeline, systems
from replycoach.generate import providers
from replycoach.io import ROOT, load_items, write_jsonl


@pytest.fixture
def demo_test(monkeypatch, fake_encoder, tmp_path):
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
    app = AppTest.from_file(str(ROOT / "src/replycoach/demo_app.py"), default_timeout=20).run()
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


def test_demo_identifies_pilot_drafts_without_loading_model(demo_test, fake_encoder):
    assert not demo_test.exception
    captions = " ".join(caption.value for caption in demo_test.caption)
    assert "Pilot examples (drafts)" in captions
    assert "20 examples" in captions
    assert not fake_encoder.calls
    assert demo_test.button(key="compare").label == "Compare reply suggestions"


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
    assert any('preview-composer selected' in block.value and selected in block.value for block in app.markdown)
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


def test_bank_switch_clears_old_suggestions_and_blocks_invalid_path(demo_test):
    app = compare_in_app(demo_test)
    app.button(key="select_bm25_standard").click().run()
    app.selectbox(key="bank_choice").select("Reviewed development file").run()
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


def enable_all_models(app, monkeypatch, stub_generator):
    from replycoach import all_models
    from replycoach.shared_ranker import PairScore

    assert 'All models (experimental)' in app.selectbox(key='system_mode').options

    class Scorer:
        identity = {'model': 'test-only', 'revision': 'fixed'}

        def score_pairs(self, pairs):
            return [PairScore(sum(map(ord, reply)) / 1000) for _, reply in pairs]

    def generator(provider):
        if provider == 'v4':
            return stub_generator
        from test_all_models import Generator
        return Generator('V5')

    monkeypatch.setattr(demo_resources, 'get_generator', generator)
    monkeypatch.setattr(demo_resources, 'get_shared_ranker', lambda selector='pretrained': Scorer(), raising=False)
    monkeypatch.setattr(all_models, 'log_event', lambda *args, **kwargs: None)
    app.selectbox(key='system_mode').select('All models (experimental)').run()
    return app


def test_all_models_shows_eight_proposals_and_click_only_fills_composer(demo_test, monkeypatch, stub_generator):
    app = enable_all_models(demo_test, monkeypatch, stub_generator)
    assert app.button(key='compare').label == 'Suggest two replies'
    compare_in_app(app)
    assert not app.exception
    assert any('4/4 models compared' in caption.value for caption in app.caption)
    assert suggestion_keys(app) == {'select_all_standard', 'select_all_local'}
    assert len(app.dataframe[0].value) == 8
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
        if provider == 'v5':
            raise RuntimeError('V5 checkpoint unavailable')
        return stub_generator

    monkeypatch.setattr(demo_resources, 'get_generator', generator)
    compare_in_app(app)
    assert not app.exception
    assert any('3/4 models available' in warning.value for warning in app.warning)
    assert any('V5 checkpoint unavailable' in warning.value for warning in app.warning)
    assert len(app.dataframe[0].value) == 6
    assert 'V5' not in set(app.dataframe[0].value['Model'])


def test_all_models_ranker_error_preserves_proposals_and_retry_recovers(demo_test, monkeypatch, stub_generator):
    app = enable_all_models(demo_test, monkeypatch, stub_generator)
    good = demo_resources.get_shared_ranker

    def unavailable(selector='pretrained'):
        raise RuntimeError('Ranker weights unavailable')

    monkeypatch.setattr(demo_resources, 'get_shared_ranker', unavailable)
    compare_in_app(app)
    assert not app.exception
    assert not suggestion_keys(app)
    assert len(app.dataframe[0].value) == 8
    assert any('Ranker weights unavailable' in error.value for error in app.error)
    monkeypatch.setattr(demo_resources, 'get_shared_ranker', good)
    app.button(key='retry_all_models').click().run()
    assert not app.exception and not app.error
    assert suggestion_keys(app) == {'select_all_standard', 'select_all_local'}


def test_reviewed_ranker_is_default_and_switching_selectors_clears_results(demo_test, monkeypatch, stub_generator):
    app = enable_all_models(demo_test, monkeypatch, stub_generator)
    assert app.selectbox(key='shared_selector_checkpoint').value == 'Fine-tuned V6 (20 reviews)'
    scorer_factory = demo_resources.get_shared_ranker
    calls = []
    def selected_scorer(selector='pretrained'):
        calls.append(selector)
        return scorer_factory(selector)
    monkeypatch.setattr(demo_resources, 'get_shared_ranker', selected_scorer)
    compare_in_app(app)
    assert calls == ['reviewed']
    for label, selector in [('Frozen MiniLM', 'pretrained'), ('Fine-tuned V6 (original)', 'finetuned'),
                            ('Fine-tuned V6 (20 reviews)', 'reviewed')]:
        app.button(key='select_all_standard').click().run()
        app.selectbox(key='shared_selector_checkpoint').select(label).run()
        assert app.session_state.last_all_models_result is None
        assert app.session_state.selected_reply == ''
        compare_in_app(app)
        assert not app.exception and calls[-1] == selector
        assert suggestion_keys(app) == {'select_all_standard', 'select_all_local'}


def test_legacy_frozen_session_does_not_override_reviewed_default(demo_test, monkeypatch, stub_generator):
    demo_test.session_state['shared_selector'] = 'Frozen MiniLM'
    app = enable_all_models(demo_test, monkeypatch, stub_generator)
    assert app.selectbox(key='shared_selector_checkpoint').value == 'Fine-tuned V6 (20 reviews)'
