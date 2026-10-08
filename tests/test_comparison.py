import pytest

from replycoach import pipeline
from replycoach.fallback import FALLBACK_LOCAL, FALLBACK_STANDARD
from replycoach.io import load_items
from replycoach.rerank import overlap
from replycoach.retrieval.v0 import V0Retriever
from replycoach.retrieval.v1 import retrieval_query
from replycoach.schema import Candidate


class RecordingRetriever:
    def __init__(self, score=0.8, error=None):
        self.calls = []
        self.score = score
        self.error = error

    def retrieve(self, message, context=None, top_k=3, source="V0", **kwargs):
        self.calls.append((message, list(context or []), top_k))
        if self.error:
            raise RuntimeError(self.error)
        return [
            Candidate(context=[], message=message, candidate_text=text, style=style,
                      source=source, score=self.score, item_id="SAVED-001")
            for style, text in (("STANDARD", "Yes, lunch sounds good."), ("LOCAL", "Can, lunch sounds good lah."))
        ]


def compare(bm25, minilm_factory, mode="M1", message="makan later?", session=None):
    return pipeline.compare_replies(
        message, session or ["Unique old turn", "meet at library"], mode,
        bm25, minilm_factory, log=False,
    )


def test_methods_receive_identical_query_memory_and_depth(monkeypatch):
    actual_select = pipeline.select_memory
    selections = []

    def select_once(*args, **kwargs):
        result = actual_select(*args, **kwargs)
        selections.append(result)
        return result

    monkeypatch.setattr(pipeline, "select_memory", select_once)
    bm25 = RecordingRetriever()
    minilm = RecordingRetriever()
    comparison = compare(bm25, lambda: minilm)
    assert len(selections) == 1
    assert bm25.calls == minilm.calls == [
        (retrieval_query("makan later?"), [retrieval_query(t) for t in selections[0]], 3)
    ]
    assert comparison.prepared.memory == selections[0]
    assert comparison.prepared.normalised_message == "eat later"
    assert comparison.prepared.detected_terms[0]["surface"] == "makan"


def test_four_suggestions_keep_original_input_and_retrieval_origin():
    comparison = compare(RecordingRetriever(), lambda: RecordingRetriever())
    assert set(comparison.results) == {"bm25", "minilm"}
    assert not comparison.errors
    for method, result in comparison.results.items():
        assert result.method == method
        assert result.memory == comparison.prepared.memory
        assert result.standard.style == "STANDARD"
        assert result.local.style == "LOCAL"
        for candidate in (result.standard, result.local):
            assert candidate.source == "V6"
            assert candidate.item_id == "SAVED-001"
            assert candidate.context == comparison.prepared.memory
            assert candidate.message == "makan later?"
            assert candidate.normalised_message == "eat later"
            assert candidate.meta["retrieval_source"] == ("V0" if method == "bm25" else "V2")
            assert candidate.meta["retrieval_method"] == method
            assert candidate.score == pytest.approx(
                0.6 * 0.8 + 0.4 * overlap(" ".join(result.memory + [result.message]), candidate.candidate_text)
            )


def test_minilm_failure_retains_real_bm25_result():
    def unavailable():
        raise RuntimeError("model unavailable")

    comparison = compare(V0Retriever.from_items(load_items()), unavailable)
    assert set(comparison.results) == {"bm25"}
    assert comparison.errors == {"minilm": "model unavailable"}
    assert comparison.results["bm25"].standard.item_id.startswith("PILOT-")


def test_minilm_encoding_failure_does_not_impersonate_bm25():
    comparison = compare(RecordingRetriever(), lambda: RecordingRetriever(error="encoding failed"))
    assert set(comparison.results) == {"bm25"}
    assert comparison.errors == {"minilm": "encoding failed"}


def test_bm25_failure_does_not_block_minilm():
    comparison = compare(RecordingRetriever(error="word index unavailable"), lambda: RecordingRetriever())
    assert set(comparison.results) == {"minilm"}
    assert comparison.errors == {"bm25": "word index unavailable"}


def test_bm25_index_load_failure_does_not_block_minilm():
    def unavailable():
        raise RuntimeError("index load failed")

    comparison = compare(unavailable, lambda: RecordingRetriever())
    assert set(comparison.results) == {"minilm"}
    assert comparison.errors == {"bm25": "index load failed"}


def test_same_low_ranking_fallback_policy_applies_to_both_methods():
    comparison = compare(RecordingRetriever(score=0), lambda: RecordingRetriever(score=0),
                         mode="M0", message="zxqv unknown")
    for result in comparison.results.values():
        assert result.standard.source == result.local.source == "fallback"
        assert result.standard.candidate_text == FALLBACK_STANDARD
        assert result.local.candidate_text == FALLBACK_LOCAL
        assert result.standard.item_id is None


def test_existing_single_method_pipeline_does_not_duplicate_memory():
    retriever = RecordingRetriever()
    coach = pipeline.ReplyCoach(retriever=retriever, memory_mode="M1")
    result = coach.suggest("makan later?", session=["Unique old turn"], log=False)
    assert retriever.calls == [(retrieval_query("makan later?"), [retrieval_query("Unique old turn")], 3)]
    assert result.method == "bm25"
    assert result.standard.context == ["Unique old turn"]
    assert result.standard.message == "makan later?"
