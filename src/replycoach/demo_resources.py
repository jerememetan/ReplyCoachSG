"""Bounded, content-keyed demo resources; MiniLM is loaded only on demand."""

import json
import os
from pathlib import Path

import streamlit as st

from replycoach.retrieval.v0 import V0Retriever
from replycoach.schema import DialogueItem
from replycoach.generate.providers import load_generator

_generators = {}
_model_group = 'baseline'


def runtime_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_model_group(group):
    """Release other model groups before loading a large transformer on an 8 GB GPU."""
    global _model_group
    if group == _model_group:
        return
    for generator in _generators.values():
        if callable(getattr(generator, 'close', None)):
            generator.close()
    generator = None
    _generators.clear()
    for resource in [get_generator, get_shared_ranker, get_minilm, get_minilm_model]:
        clear = getattr(resource, 'clear', None)
        if clear is not None:
            clear()
    import gc
    gc.collect()
    if runtime_device() == 'cuda':
        import torch
        torch.cuda.empty_cache()
    _model_group = group


@st.cache_resource(max_entries=1, show_spinner=False)
def get_minilm_model():
    from sentence_transformers import SentenceTransformer
    from replycoach.retrieval.v2 import MODEL_NAME

    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(max_entries=4, show_spinner=False)
def get_bm25(bank_json: str) -> V0Retriever:
    items = [DialogueItem.from_dict(row) for row in json.loads(bank_json)]
    return V0Retriever.from_items(items)


@st.cache_resource(max_entries=4, show_spinner=False)
def get_minilm(bank_json: str):
    from replycoach.retrieval.v2 import V2Retriever

    items = [DialogueItem.from_dict(row) for row in json.loads(bank_json)]
    return V2Retriever.from_items(items, model=get_minilm_model(), cache_file=None)


@st.cache_resource(max_entries=4, show_spinner=False)
def get_generator(provider: str):
    """Generator resources, loaded only after a generation request."""
    generator = load_generator(provider)
    if provider == 'v4' and hasattr(generator, 'model'):
        import torch
        generator.device = torch.device(runtime_device())
        generator.model.to(generator.device)
    _generators[provider] = generator
    return generator


@st.cache_resource(max_entries=4, show_spinner=False)
def get_shared_ranker(selector='pretrained', *, device=None):
    from replycoach.shared_ranker import CrossEncoderScorer
    from replycoach.io import ROOT
    device = device or runtime_device()
    if selector == 'qwen_dailydialog':
        from replycoach.qwen_reranker import QwenReranker

        base = Path(os.environ.get('REPLYCOACH_QWEN_BASE', str(ROOT/'artifacts/reranker_qwen/pretrained')))
        adapter = Path(os.environ.get(
            'REPLYCOACH_QWEN_DD_ADAPTER',
            str(ROOT/'artifacts/reranker_qwen/dd_result_20261029/stage2_reviewed/best'),
        ))
        for path, variable in [(base, 'REPLYCOACH_QWEN_BASE'), (adapter, 'REPLYCOACH_QWEN_DD_ADAPTER')]:
            if not path.is_dir():
                raise FileNotFoundError(f'Qwen checkpoint folder missing: {path}. Supply the private files or set {variable}.')
        return QwenReranker(base, adapter=adapter, device=device)
    if selector == 'pretrained':
        return CrossEncoderScorer(device=device)
    if selector == 'finetuned':
        checkpoint = Path(os.environ.get('REPLYCOACH_V6_CHECKPOINT', str(ROOT/'artifacts/reranker_v6/best')))
        return CrossEncoderScorer(checkpoint=checkpoint, device=device)
    if selector == 'reviewed':
        checkpoint = Path(os.environ.get(
            'REPLYCOACH_V6_REVIEWED_CHECKPOINT',
            str(ROOT/'artifacts/reranker_v6/kaggle_output_20261027_reviewed01/reranker_v6/best'),
        ))
        return CrossEncoderScorer(checkpoint=checkpoint, device=device)
    raise ValueError('unknown shared selector')
