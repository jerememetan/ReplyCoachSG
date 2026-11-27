"""Session memory: M0 none, M1 last 3, M2 newest 2 plus up to 2 relevant older turns."""

from __future__ import annotations

from replycoach.retrieval import build_index


WINDOW = 8
NEWEST = 2
RELEVANT = 2
LAST3 = 3


def last_window(session: list[str], window: int = WINDOW) -> list[str]:
    return [turn for turn in session[-window:] if str(turn).strip()]


def m0_no_memory(session: list[str], message: str) -> list[str]:
    return []


def m1_last3(session: list[str], message: str) -> list[str]:
    return last_window(session, LAST3)


def m2_dynamic(session: list[str], message: str, newest: int = NEWEST, relevant: int = RELEVANT) -> list[str]:
    """Keep the newest turns; add up to two BM25-relevant older turns from the last eight."""
    recent = last_window(session, WINDOW)
    if not recent:
        return []
    newest_turns = recent[-newest:]
    older = recent[:-newest] if len(recent) > newest else []
    if not older or relevant <= 0:
        return newest_turns
    index = build_index(older)
    picked_idx = [idx for idx, score in index.rank(message, top_k=relevant) if score > 0]
    picked_idx.sort()
    selected_older = [older[idx] for idx in picked_idx]
    chronological = [turn for turn in recent if turn in selected_older or turn in newest_turns]
    return chronological


MEMORY_FNS = {
    "M0": m0_no_memory,
    "M1": m1_last3,
    "M2": m2_dynamic,
}


def select_memory(session: list[str], message: str, mode: str = "M2") -> list[str]:
    if mode not in MEMORY_FNS:
        raise ValueError(f"unknown memory mode {mode}")
    return MEMORY_FNS[mode](session, message)
