from pathlib import Path
from threading import Event, Thread

import pytest

from replycoach.authoring import load_author_items, save_author_item
from replycoach.io import save_items
from replycoach.schema import DialogueItem, Message, Replies, Turn
from replycoach import team_review_app
from replycoach.team_review_app import save_review


def make_item(item_id: str = "ITEM-B-001") -> DialogueItem:
    return DialogueItem(
        id=item_id,
        split="unassigned",
        scenario="meals_and_food",
        intent="invite_or_propose",
        session_memory=[Turn(speaker="peer", text="Free later?")],
        message=Message(text="Dinner?"),
        replies=Replies(standard="Yes, let's go.", local="Can, let's go."),
        author="Author B",
        notes="Check the tone.",
        second_reviewer="Legacy second",
        second_reviewer_notes="Legacy note",
    )


def test_save_review_persists_structured_edits_and_keeps_legacy_metadata(tmp_path: Path):
    path = tmp_path / "items_B.jsonl"
    save_items(path, [make_item()])
    items = load_author_items(path)

    save_review(
        path, items, 0, " Reviewer A ",
        expected_snapshot=items[0].to_dict(),
        scenario="study_sessions_and_spaces",
        intent="request_information_or_status",
        conversation="peer: Is the library open?\nuser: I think so.",
        message="Which floor?",
        language_mix=["en"],
        local_terms="library, floor",
        standard="The second floor.",
        local="Second floor lah.",
        privacy_ok=True,
        reviewer_notes="Verified wording.",
    )

    saved = load_author_items(path)[0]
    assert saved.reviewer == "Reviewer A"
    assert saved.status == "completed"
    assert saved.scenario == "study_sessions_and_spaces"
    assert saved.intent == "request_information_or_status"
    assert saved.session_memory[1].text == "I think so."
    assert saved.message.text == "Which floor?"
    assert saved.message.local_terms == ["library", "floor"]
    assert saved.replies.local == "Second floor lah."
    assert saved.reviewer_notes == "Verified wording."
    assert saved.author == "Author B"
    assert saved.notes == "Check the tone."
    assert saved.split == "unassigned"
    assert saved.second_reviewer == "Legacy second"
    assert saved.second_reviewer_notes == "Legacy note"


def test_save_review_blocks_different_reviewer_without_changing_file(tmp_path: Path):
    path = tmp_path / "items_B.jsonl"
    item = make_item()
    item.reviewer = "Reviewer A"
    item.status = "completed"
    save_items(path, [item])
    original = path.read_bytes()
    opened_items = load_author_items(path)

    with pytest.raises(ValueError, match="already reviewed"):
        save_review(
            path, opened_items, 0, "Reviewer C",
            expected_snapshot=opened_items[0].to_dict(),
            scenario="meals_and_food",
            intent="invite_or_propose",
            conversation="peer: Free later?",
            message="Dinner?",
            language_mix=["en"],
            local_terms="",
            standard="Yes, let's go.",
            local="Can, let's go.",
            privacy_ok=True,
            reviewer_notes="Takeover",
        )

    assert path.read_bytes() == original


@pytest.mark.parametrize("row", ['{}\n', '{"id":"ITEM-B-001","message":null,"replies":{}}\n', '[]\n'])
def test_malformed_assigned_rows_return_read_error_without_changing_file(tmp_path: Path, row: str):
    path = tmp_path / "items_B.jsonl"
    path.write_text(row, encoding="utf-8")

    items, error = team_review_app.load_review_items(path)

    assert items == []
    assert error.startswith(f"Could not read assigned file {path.name}:")
    assert path.read_text(encoding="utf-8") == row


def test_save_review_rejects_changed_selected_item_without_overwriting_it(tmp_path: Path):
    path = tmp_path / "items_B.jsonl"
    save_items(path, [make_item()])
    opened_items = load_author_items(path)
    snapshot = opened_items[0].to_dict()
    changed_item = load_author_items(path)[0]
    changed_item.message.text = "Updated by author"
    save_items(path, [changed_item])
    current_bytes = path.read_bytes()

    with pytest.raises(ValueError, match="changed since you opened it; reload"):
        save_review(
            path, opened_items, 0, "Reviewer A",
            expected_snapshot=snapshot,
            scenario="meals_and_food", intent="invite_or_propose",
            conversation="peer: Free later?", message="Dinner?",
            language_mix=["en"], local_terms="",
            standard="Yes, let's go.", local="Can, let's go.",
            privacy_ok=True, reviewer_notes="Reviewed",
        )

    assert path.read_bytes() == current_bytes


def test_save_review_preserves_concurrent_change_to_another_row(tmp_path: Path):
    path = tmp_path / "items_B.jsonl"
    save_items(path, [make_item(), make_item("ITEM-B-002")])
    opened_items = load_author_items(path)
    snapshot = opened_items[0].to_dict()
    latest_items = load_author_items(path)
    latest_items[1].notes = "Edited by author after reviewer opened first item"
    save_items(path, latest_items)

    save_review(
        path, opened_items, 0, "Reviewer A",
        expected_snapshot=snapshot,
        scenario="meals_and_food", intent="invite_or_propose",
        conversation="peer: Free later?", message="Dinner?",
        language_mix=["en"], local_terms="",
        standard="Yes, let's go.", local="Can, let's go.",
        privacy_ok=True, reviewer_notes="Reviewed",
    )

    saved = load_author_items(path)
    assert saved[0].reviewer == "Reviewer A"
    assert saved[1].notes == "Edited by author after reviewer opened first item"


def test_review_and_author_saves_serialize_the_entire_update(tmp_path: Path, monkeypatch):
    path = tmp_path / "items_B.jsonl"
    draft = make_item("ITEM-B-002")
    draft.second_reviewer = ""
    draft.second_reviewer_notes = ""
    save_items(path, [make_item(), draft])
    opened_items = load_author_items(path)
    review_reached_write = Event()
    release_review_write = Event()
    author_finished = Event()
    errors = []
    original_save_items = team_review_app.save_items

    def delayed_review_write(*args):
        review_reached_write.set()
        if not release_review_write.wait(5):
            raise TimeoutError("review test coordination timed out")
        original_save_items(*args)

    monkeypatch.setattr(team_review_app, "save_items", delayed_review_write)

    def run_review():
        try:
            save_review(
                path, opened_items, 0, "Reviewer A",
                expected_snapshot=opened_items[0].to_dict(),
                scenario="meals_and_food", intent="invite_or_propose",
                conversation="peer: Free later?", message="Dinner?",
                language_mix=["en"], local_terms="",
                standard="Yes, let's go.", local="Can, let's go.",
                privacy_ok=True, reviewer_notes="Reviewed",
            )
        except Exception as exc:
            errors.append(exc)

    def run_author():
        try:
            draft.notes = "Edited while review was saving"
            save_author_item(path, draft)
        except Exception as exc:
            errors.append(exc)
        finally:
            author_finished.set()

    review_thread = Thread(target=run_review)
    author_thread = Thread(target=run_author)
    review_thread.start()
    try:
        assert review_reached_write.wait(5)
        author_thread.start()
        assert not author_finished.wait(0.25), "author save must wait for review write"
    finally:
        release_review_write.set()
        review_thread.join(5)
        if author_thread.ident is not None:
            author_thread.join(5)

    assert not review_thread.is_alive()
    assert not author_thread.is_alive()
    assert errors == []
    saved = load_author_items(path)
    assert saved[0].reviewer == "Reviewer A"
    assert saved[1].notes == "Edited while review was saving"
