from replycoach.demo_app import build_preview_html, parse_preview_line


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
