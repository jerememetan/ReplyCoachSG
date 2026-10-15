import numpy as np
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

from replycoach.schema import Candidate


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


class StubGenerator:
    """Test-only local generator; never registered in the application."""

    source = "V4"

    def __init__(self):
        self.calls = []
        self.outputs = [
            Candidate([], "", "Yes, lunch at the library sounds good.", "STANDARD", "V4", 0.8),
            Candidate([], "", "Can, lunch then library lah.", "LOCAL", "V4", 0.8),
        ]

    def generate(self, message, context):
        self.calls.append((message, list(context)))
        return self.outputs


@pytest.fixture
def stub_generator():
    return StubGenerator()
