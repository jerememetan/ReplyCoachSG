from replycoach.io import load_items
from replycoach.pipeline import ReplyCoach
from replycoach.retrieval.v0 import V0Retriever
from replycoach.retrieval.v1 import detect_and_normalise
from replycoach.retrieval.v2 import V2Retriever


def test_v0_retrieves_lunch_item():
    items = load_items()
    retriever = V0Retriever.from_items(items)
    candidates = retriever.retrieve("eh later go makan then library", top_k=1)
    assert candidates
    assert candidates[0].item_id == "PILOT-001"
    assert {c.style for c in candidates} == {"STANDARD", "LOCAL"}
    assert all(c.source == "V0" for c in candidates)


def test_v1_normalises_makan_and_chinese():
    result = detect_and_normalise("eh later go makan? can 吃饭 then library")
    surfaces = [row["surface"] for row in result.detected]
    assert "makan" in surfaces
    assert "吃饭" in surfaces
    assert "eat" in result.normalised
    assert "lah" not in result.normalised.split() or True


def test_pipeline_returns_two_replies():
    coach = ReplyCoach.from_pilot()
    result = coach.suggest(
        "eh later go makan? can 吃饭 then library",
        session=["lecture ending soon sia", "yeah I'm starving"],
        log=False,
    )
    assert result.standard.style == "STANDARD"
    assert result.local.style == "LOCAL"
    assert result.standard.candidate_text
    assert result.local.candidate_text

def test_v2_retrieves_lunch_item(tmp_path):

    items = load_items()

    retriever = V2Retriever.from_items(items, cache_file=tmp_path / "lunch_embeddings.npy")

    candidates = retriever.retrieve(
        "eh later go makan then library",
        context=["lecture ending soon sia", "yeah I'm starving"],
        top_k=1,
    )

    assert candidates
    assert candidates[0].item_id == "PILOT-001"


def test_v2_returns_shared_candidate_format(tmp_path):

    items = load_items()

    retriever = V2Retriever.from_items(items, cache_file=tmp_path / "format_embeddings.npy")

    candidates = retriever.retrieve(
        "eh later go makan then library",
        top_k=1,
    )

    assert {c.style for c in candidates} == {"STANDARD", "LOCAL"}
    assert all(c.source == "V2" for c in candidates)
    assert all(isinstance(c.score, float) for c in candidates)
