"""Bounded, content-keyed demo resources; MiniLM is loaded only on demand."""

import json

import streamlit as st

from replycoach.retrieval.v0 import V0Retriever
from replycoach.schema import DialogueItem


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
