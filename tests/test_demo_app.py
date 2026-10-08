from replycoach.demo_app import build_preview_html, parse_preview_line

import pytest
from streamlit.testing.v1 import AppTest

from replycoach import demo_resources, pipeline
from replycoach.io import ROOT, load_items, write_jsonl


@pytest.fixture
def demo_test(monkeypatch, fake_encoder):
    cached_minilm = demo_resources.get_minilm
    demo_resources.get_bm25.clear()
    cached_minilm.clear()
    monkeypatch.setattr(demo_resources, "get_minilm_model", lambda: fake_encoder)
    # Keep AppTest suggestions from appending to the team's experiment log.
    monkeypatch.setattr(pipeline, "log_event", lambda *args, **kwargs: None)
    app = AppTest.from_file(str(ROOT / "src/replycoach/demo_app.py"), default_timeout=20).run()
    yield app
    demo_resources.get_bm25.clear()
    cached_minilm.clear()


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
