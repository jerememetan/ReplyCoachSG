"""Local generator handoff points. No smoke models or synthetic providers."""

from typing import Protocol
import os
from pathlib import Path

from replycoach.schema import Candidate


class GeneratorUnavailable(RuntimeError):
    """A trained local generator has not been supplied or cannot be loaded."""


class ReplyGenerator(Protocol):
    source: str  # V4 or V5; kept on Candidates before the shared reranker.

    def generate(self, message: str, context: list[str]) -> list[Candidate]:
        """Return both STANDARD and LOCAL, with finite input scores in [0, 1]."""
        ...


def load_v4() -> ReplyGenerator:
    from replycoach.generate.rnn_provider import RNNGenerator
    from replycoach.io import ROOT

    path = Path(os.environ.get("REPLYCOACH_V4_CHECKPOINT", str(ROOT / "artifacts/rnn_v4_task.pt")))
    if not path.is_file():
        raise GeneratorUnavailable(
            f"Role C: trained V4 checkpoint missing: {path}. Supply the task checkpoint "
            "or set REPLYCOACH_V4_CHECKPOINT; see roles/C_rnn_attention/HANDOFF.md."
        )
    try:
        generator = RNNGenerator(path)
    except (KeyError, ValueError) as exc:
        raise GeneratorUnavailable(f"Role C: invalid task-trained V4 checkpoint: {exc}") from exc
    if generator.source != "V4":
        raise GeneratorUnavailable("Role C: V4 provider requires attention weights, not V3 weights")
    return generator


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
