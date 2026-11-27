"""V6 baseline: combine retrieval score with token overlap against message + memory."""

from __future__ import annotations

from replycoach.schema import Candidate
from replycoach.tokenize import tokenize


def overlap(query: str, text: str) -> float:
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


def rerank(candidates: list[Candidate], message: str, context: list[str] | None = None) -> list[Candidate]:
    # ##TODO: FOR ROLE B — validate cross-source score scales with C/D on dev.
    # A [0, 1] range is not calibration. Keep the baseline until the experiment
    # supports a change; never tune weights on the locked test set.
    memory = " ".join(context or [])
    query = f"{memory} {message}".strip()
    rescored = []
    for candidate in candidates:
        sim = overlap(query, candidate.candidate_text)
        new_score = 0.6 * float(candidate.score) + 0.4 * sim
        meta = dict(candidate.meta)
        meta["v6_overlap"] = round(sim, 4)
        meta["v6_input_score"] = float(candidate.score)
        rescored.append(
            Candidate(
                context=candidate.context,
                message=candidate.message,
                candidate_text=candidate.candidate_text,
                style=candidate.style,
                source="V6",
                score=float(new_score),
                item_id=candidate.item_id,
                normalised_message=candidate.normalised_message,
                meta=meta,
            )
        )
    rescored.sort(key=lambda item: (item.style, -item.score))
    return rescored


def pick_two(candidates: list[Candidate]) -> dict[str, Candidate]:
    """One STANDARD and one LOCAL, highest score each."""
    chosen: dict[str, Candidate] = {}
    for style in ("STANDARD", "LOCAL"):
        styled = [c for c in candidates if c.style == style]
        if styled:
            chosen[style] = max(styled, key=lambda c: c.score)
    return chosen
