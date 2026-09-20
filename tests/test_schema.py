from replycoach.io import load_items
from replycoach.schema import Candidate, DialogueItem


def test_pilot_loads_and_validates():
    items = load_items()
    assert len(items) == 20
    ids = [item.id for item in items]
    assert len(ids) == len(set(ids))
    for item in items:
        assert item.split == "pilot"
        assert item.privacy_ok
        assert item.replies.standard
        assert item.replies.local
        assert item.replies.standard != item.replies.local or "English-only" in item.notes


def test_candidate_roundtrip():
    candidate = Candidate(
        context=["lecture ending soon sia"],
        message="eh later go makan?",
        candidate_text="Want to grab lunch first?",
        style="STANDARD",
        source="V0",
        score=1.23,
        item_id="PILOT-001",
    )
    restored = Candidate.from_dict(candidate.to_dict())
    assert restored.candidate_text == candidate.candidate_text
    assert restored.style == "STANDARD"


def test_dialogue_item_preserves_two_reviewer_notes():
    item = DialogueItem.from_dict(
        {
            "id": "X",
            "split": "pilot",
            "scenario": "lunch",
            "intent": "propose_meal",
            "session_memory": [],
            "message": {"text": "hello"},
            "replies": {"standard": "Hello.", "local": "Hello."},
            "reviewer_notes": "First pass",
            "second_reviewer_notes": "Second pass",
        }
    )
    restored = DialogueItem.from_dict(item.to_dict())
    assert restored.reviewer_notes == "First pass"
    assert restored.second_reviewer_notes == "Second pass"


def test_unassigned_authoring_item_is_a_valid_draft():
    item = DialogueItem.from_dict(
        {
            "id": "ITEM-E-001",
            "split": "unassigned",
            "scenario": "meal",
            "intent": "invite_or_propose",
            "session_memory": [{"speaker": "peer", "text": "Free after class?"}],
            "message": {"text": "Want to get dinner?", "language_mix": ["en"], "local_terms": []},
            "replies": {"standard": "Sure, where should we meet?", "local": "Can, where should we meet?"},
            "author": "Jereme",
            "privacy_ok": True,
        }
    )

    assert item.split == "unassigned"
