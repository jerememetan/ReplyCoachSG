"""Local generator handoff points. No smoke models or synthetic providers."""

from typing import Protocol

from replycoach.schema import Candidate


class GeneratorUnavailable(RuntimeError):
    """A trained local generator has not been supplied or cannot be loaded."""


class ReplyGenerator(Protocol):
    source: str  # V4 or V5; kept on Candidates before the shared reranker.

    def generate(self, message: str, context: list[str]) -> list[Candidate]:
        """Return both STANDARD and LOCAL, with finite input scores in [0, 1]."""
        ...


def load_v4() -> ReplyGenerator:
    # ##TODO: FOR ROLE C — return your trained V4 ReplyGenerator here.
    # Load the task-trained checkpoint + its vocabulary once, not a smoke model.
    # See docs/role_handoffs.md for the exact generate()/Candidate contract.
    raise GeneratorUnavailable(
        "Role C: trained V4 generator is not connected. Complete load_v4() in "
        "src/replycoach/generate/providers.py; see docs/role_handoffs.md."
    )


def load_v5() -> ReplyGenerator:
    # ##TODO: FOR ROLE D — return your trained local mT5/LoRA ReplyGenerator here.
    # Load the selected base weights, adapter (if used) and matching tokenizer.
    # No API calls. See docs/role_handoffs.md; do not use raw mT5 smoke output.
    raise GeneratorUnavailable(
        "Role D: trained V5 generator is not connected. Complete load_v5() in "
        "src/replycoach/generate/providers.py; see docs/role_handoffs.md."
    )


def load_generator(provider: str) -> ReplyGenerator:
    if provider not in {"v4", "v5"}:
        raise ValueError("generator provider must be v4 or v5")
    try:
        generator = load_v4() if provider == "v4" else load_v5()
    except GeneratorUnavailable:
        raise
    except (OSError, RuntimeError) as exc:
        owner = "C" if provider == "v4" else "D"
        raise GeneratorUnavailable(f"Role {owner}: cannot load trained {provider.upper()} model: {exc}") from exc
    expected = provider.upper()
    if getattr(generator, "source", None) != expected or not callable(getattr(generator, "generate", None)):
        raise ValueError(f"provider {provider} must return a {expected} ReplyGenerator")
    return generator
