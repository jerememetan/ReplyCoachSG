import copy
import json

import numpy as np
import pytest

from replycoach.io import load_items


def test_v2_accepts_shared_encoder_without_writing_artifacts(fake_encoder, monkeypatch):
    from replycoach.retrieval.v2 import V2Retriever

    def unexpected_write(*args, **kwargs):
        raise AssertionError("comparison must not overwrite shared disk embeddings")

    monkeypatch.setattr(np, "save", unexpected_write)
    retriever = V2Retriever.from_items(load_items(), model=fake_encoder, cache_file=None)
    candidates = retriever.retrieve("makan lunch library", top_k=2)
    assert len(candidates) == 4
    assert {candidate.style for candidate in candidates} == {"STANDARD", "LOCAL"}
    assert all(candidate.source == "V2" for candidate in candidates)
    assert all(candidate.item_id in {item.id for item in retriever.items} for candidate in candidates)
    assert retriever.model is fake_encoder
    assert retriever.embeddings.shape == (20, 256)


def test_v2_preserves_optional_explicit_disk_cache(tmp_path, fake_encoder):
    from replycoach.retrieval.v2 import V2Retriever

    path = tmp_path / "chosen.npy"
    retriever = V2Retriever.from_items(load_items(), model=fake_encoder, cache_file=path)
    assert np.array_equal(np.load(path), retriever.embeddings)


def test_cached_resources_reuse_model_and_invalidate_changed_bank(fake_encoder, monkeypatch):
    from replycoach import demo_resources
    import sentence_transformers

    constructed = []

    def load_model(name):
        constructed.append(name)
        return fake_encoder

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", load_model)
    demo_resources.get_minilm_model.clear()
    demo_resources.get_minilm.clear()
    demo_resources.get_bm25.clear()
    rows = [item.to_dict() for item in load_items()]
    bank_json = json.dumps(rows, ensure_ascii=False)
    bm25 = demo_resources.get_bm25(bank_json)
    assert not constructed  # BM25 and initial rendering do not load MiniLM.
    first = demo_resources.get_minilm(bank_json)
    assert demo_resources.get_minilm(bank_json) is first
    assert len(fake_encoder.calls) == 1
    changed_rows = copy.deepcopy(rows)
    changed_rows[0]["message"]["text"] = "zebra astronomy telescope observatory"
    changed_json = json.dumps(changed_rows, ensure_ascii=False)
    second = demo_resources.get_minilm(changed_json)
    assert second is not first
    assert second.model is first.model
    assert len(constructed) == 1
    assert len(fake_encoder.calls) == 2
    assert not np.array_equal(first.embeddings[0], second.embeddings[0])
    assert demo_resources.get_bm25(bank_json) is bm25
    assert demo_resources.get_bm25(changed_json).items[0].message.text == "zebra astronomy telescope observatory"
    demo_resources.get_minilm_model.clear()
    demo_resources.get_minilm.clear()
    demo_resources.get_bm25.clear()


def test_failed_model_load_can_be_retried(fake_encoder, monkeypatch):
    from replycoach import demo_resources
    import sentence_transformers

    attempts = []

    def load_model(name):
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("download unavailable")
        return fake_encoder

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", load_model)
    demo_resources.get_minilm_model.clear()
    demo_resources.get_minilm.clear()
    bank_json = json.dumps([load_items()[0].to_dict()])
    with pytest.raises(RuntimeError, match="download unavailable"):
        demo_resources.get_minilm(bank_json)
    recovered = demo_resources.get_minilm(bank_json)
    assert recovered.items[0].id == "PILOT-001"
    assert len(attempts) == 2
    demo_resources.get_minilm_model.clear()
    demo_resources.get_minilm.clear()


def test_selected_generator_is_cached_without_reloading_weights(monkeypatch, stub_generator):
    from replycoach import demo_resources
    from replycoach.generate import providers

    loads = []

    def load_model():
        loads.append("v4")
        return stub_generator

    monkeypatch.setattr(providers, "load_v4", load_model)
    demo_resources.get_generator.clear()
    try:
        assert demo_resources.get_generator("v4") is stub_generator
        assert demo_resources.get_generator("v4") is stub_generator
        assert loads == ["v4"]
    finally:
        demo_resources.get_generator.clear()
