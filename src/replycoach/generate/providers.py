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
