import importlib
from dataclasses import replace

import pytest

from replycoach.fallback import FALLBACK_LOCAL, FALLBACK_STANDARD
from replycoach.io import load_items
from replycoach.retrieval.v0 import V0Retriever


def systems():
    return importlib.import_module("replycoach.systems")


def run(mode, generator=None, retriever=None, message="makan later?", memory="M1"):
    return systems().run_system(
        message, ["meet at library"], memory, mode=mode,
        retrieval_method="bm25",
        retriever_factory=retriever or (lambda: V0Retriever.from_items(load_items())),
        generator_factory=generator,
    )


def should_not_load():
    raise AssertionError("Unused source must not load")


def test_retrieval_only_never_loads_generator():
    output = run("retrieval_only", should_not_load)
    assert output.effective_mode == "retrieval_only"
    assert not output.errors
    assert output.retrieval_count == 6
    assert output.generation_count == 0
    assert output.result.standard.item_id.startswith("PILOT-")


def test_generator_only_uses_original_message_and_selected_memory(stub_generator):
    output = run("generator_only", lambda: stub_generator, should_not_load)
    assert output.effective_mode == "generator_only"
    assert not output.errors
    assert stub_generator.calls == [("makan later?", ["meet at library"])]
    assert output.retrieval_count == 0
    assert output.generation_count == 2
    assert output.result.standard.candidate_text == stub_generator.outputs[0].candidate_text
    assert output.result.standard.meta["generation_source"] == "V4"
    assert "retrieval_source" not in output.result.standard.meta
    assert output.result.standard.item_id is None


def test_hybrid_combines_sources_before_ranking(stub_generator):
    output = run("hybrid", lambda: stub_generator)
    assert output.effective_mode == "hybrid"
    assert {candidate.meta["candidate_source"] for candidate in output.result.candidates} == {"V0", "V4"}
    assert len(output.result.candidates) == 8
    retrieved = [candidate for candidate in output.result.candidates if candidate.item_id]
    assert all(candidate.meta["retrieval_method"] == "bm25" for candidate in retrieved)
    assert all(candidate.message == "makan later?" for candidate in output.result.candidates)


@pytest.mark.parametrize("mode,effective", [("hybrid", "retrieval_only"), ("generator_only", "fallback_only")])
def test_missing_model_is_not_reported_as_real_generation(mode, effective):
    output = run(mode)
    assert output.requested_mode == mode
    assert output.effective_mode == effective
    assert "generator" in output.errors
    assert output.generation_count == 0
    if mode == "generator_only":
        assert output.result.standard.source == output.result.local.source == "fallback"
        assert output.result.standard.candidate_text == FALLBACK_STANDARD
        assert output.result.local.candidate_text == FALLBACK_LOCAL


def test_retrieval_failure_does_not_block_generator(stub_generator):
    output = run("hybrid", lambda: stub_generator, should_not_load)
    assert output.effective_mode == "generator_only"
    assert "retrieval" in output.errors
    assert output.result.standard.meta["generation_source"] == "V4"


@pytest.mark.parametrize("text", ["", "  ", None, "<unk>", "hello <extra_id_0>"])
def test_invalid_decoder_text_does_not_escape_to_hybrid(stub_generator, text):
    stub_generator.outputs[0] = replace(stub_generator.outputs[0], candidate_text=text)
    output = run("hybrid", lambda: stub_generator)
    assert output.effective_mode == "retrieval_only"
    assert output.generation_count == 0
    assert "generator" in output.errors
    assert all(candidate.item_id for candidate in output.result.candidates)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.2, 1.2, True, "0.9"])
def test_invalid_generator_scores_are_rejected(stub_generator, score):
    stub_generator.outputs[0] = replace(stub_generator.outputs[0], score=score)
    output = run("hybrid", lambda: stub_generator)
    assert output.effective_mode == "retrieval_only"
    assert "generator" in output.errors


@pytest.mark.parametrize("mutation", ["empty", "missing_style", "wrong_source", "saved_id", "not_candidate"])
def test_generator_contract_failures_are_explicit(stub_generator, mutation):
    if mutation == "empty":
        stub_generator.outputs = []
    elif mutation == "missing_style":
        stub_generator.outputs = stub_generator.outputs[:1]
    elif mutation == "wrong_source":
        stub_generator.outputs[0] = replace(stub_generator.outputs[0], source="V5")
    elif mutation == "saved_id":
        stub_generator.outputs[0] = replace(stub_generator.outputs[0], item_id="PILOT-001")
    else:
        stub_generator.outputs = ["not a Candidate"]
    output = run("generator_only", lambda: stub_generator)
    assert output.effective_mode == "fallback_only"
    assert "generator" in output.errors


def test_low_scoring_generation_uses_the_existing_fallback(stub_generator):
    stub_generator.outputs = [replace(candidate, score=0) for candidate in stub_generator.outputs]
    output = run("generator_only", lambda: stub_generator, message="zxqv unrelated", memory="M0")
    assert not output.errors
    assert output.result.standard.source == output.result.local.source == "fallback"


def test_three_mode_comparison_reuses_one_generation_and_one_retrieval(stub_generator):
    retriever = V0Retriever.from_items(load_items())
    retrieval_calls = []

    class CountedRetriever:
        def retrieve(self, message, **kwargs):
            retrieval_calls.append((message, kwargs["context"]))
            return retriever.retrieve(message, **kwargs)

    comparison = systems().compare_system_modes(
        "makan later?", ["meet at library"], "M1", retrieval_method="bm25",
        retriever_factory=lambda: CountedRetriever(), generator_factory=lambda: stub_generator,
    )
    assert set(comparison.results) == {"retrieval_only", "generator_only", "hybrid"}
    assert len(retrieval_calls) == len(stub_generator.calls) == 1
    for output in comparison.results.values():
        assert output.prepared is comparison.prepared
        assert not output.errors
    generation_texts = {candidate.candidate_text for candidate in comparison.results["generator_only"].result.candidates}
    hybrid_generation_texts = {
        candidate.candidate_text for candidate in comparison.results["hybrid"].result.candidates
        if candidate.meta["candidate_source"] == "V4"
    }
    assert generation_texts == hybrid_generation_texts


def test_invalid_mode_is_rejected_before_sources_load():
    with pytest.raises(ValueError, match="mode"):
        run("not_a_mode", should_not_load, should_not_load)


@pytest.mark.parametrize("meta", [{"tensor": object()}, {"bad": float("nan")}, None])
def test_non_json_generation_metadata_cannot_break_ui_or_export(stub_generator, meta):
    stub_generator.outputs[0] = replace(stub_generator.outputs[0], meta=meta)
    output = run("hybrid", lambda: stub_generator)
    assert output.effective_mode == "retrieval_only"
    assert "generator" in output.errors
