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
    # The trained mT5 + LoRA checkpoint in $V5_CHECKPOINT, else the default in generate/v5_provider.py.
    try:
        from replycoach.generate.v5_provider import CHECKPOINT_ENV, V5Generator, configured_checkpoint
    except ImportError as exc:
        raise GeneratorUnavailable(f"Role D: V5 needs torch, transformers and peft (the mt5 extras): {exc}") from exc
    checkpoint = configured_checkpoint()
    if not (checkpoint / "v5_input.json").exists():
        raise GeneratorUnavailable(
            f"Role D: no trained V5 checkpoint at {checkpoint}. Copy one into artifacts/ or set {CHECKPOINT_ENV} "
            "to a stage 2 checkpoint folder, e.g. artifacts/spk1_ul025_s425/best."
        )
    from replycoach.generate.hybrid_v5 import configured_rewriter, HybridV5Generator, LOCAL_REWRITE_ENV
    rewriter = configured_rewriter()
    if rewriter is not None:
        if not (rewriter / 'v5_input.json').exists():
            raise GeneratorUnavailable(f'No LOCAL rewrite checkpoint at {rewriter}; check {LOCAL_REWRITE_ENV}.')
        return HybridV5Generator(checkpoint, rewriter)
    return V5Generator(checkpoint)


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
