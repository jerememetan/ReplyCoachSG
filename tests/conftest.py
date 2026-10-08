import numpy as np
import pytest
from sklearn.feature_extraction.text import HashingVectorizer


class DeterministicEncoder:
    """Normalized lexical vectors isolate retrieval logic from model downloads."""

    def __init__(self):
        self.calls = []
        self.vectorizer = HashingVectorizer(n_features=256, alternate_sign=False, norm="l2")

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.asarray(self.vectorizer.transform(texts).toarray(), dtype=np.float32)


@pytest.fixture
def fake_encoder():
    return DeterministicEncoder()
