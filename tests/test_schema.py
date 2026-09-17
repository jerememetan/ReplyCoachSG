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
