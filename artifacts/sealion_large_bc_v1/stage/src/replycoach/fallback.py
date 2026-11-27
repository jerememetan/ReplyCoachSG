"""Safe fallbacks when retrieval/generation confidence is low."""

from __future__ import annotations

from replycoach.schema import Candidate

FALLBACK_STANDARD = "Got it. Would you like to confirm a time and place?"
FALLBACK_LOCAL = "Okay, want to confirm the time and place?"
MIN_SCORE = 0.15


def fallback_candidate(message: str, context: list[str], style: str) -> Candidate:
    text = FALLBACK_STANDARD if style == "STANDARD" else FALLBACK_LOCAL
    candidate = Candidate(
        context=context,
        message=message,
        candidate_text=text,
        style=style,  # type: ignore[arg-type]
        source="fallback",
        score=0.0,
        meta={"reason": "low_confidence_or_unknown"},
    )
    candidate.validate()
    return candidate


def maybe_fallback(candidate: Candidate | None, message: str, context: list[str], style: str) -> Candidate:
    if candidate is None or candidate.score < MIN_SCORE:
        return fallback_candidate(message, context, style)
    return candidate
