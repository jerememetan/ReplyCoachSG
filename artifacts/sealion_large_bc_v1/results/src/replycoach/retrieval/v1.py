"""V1: detect local terms and normalise them for retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from replycoach.lexicon import LocalTerm, load_lexicon, longest_first
from replycoach.tokenize import tokenize, tokenize_join


@dataclass
class Normalisation:
    original: str
    normalised: str
    detected: list[dict]


def detect_and_normalise(text: str, terms: list[LocalTerm] | None = None) -> Normalisation:
    lexicon = longest_first(terms or load_lexicon())
    tokens = tokenize(text)
    detected: list[dict] = []
    used = [False] * len(tokens)
    canonical_tokens: list[str] = []
    index = 0
    while index < len(tokens):
        matched = None
        for term in lexicon:
            surface = tokenize(term.surface)
            if not surface:
                continue
            end = index + len(surface)
            if end <= len(tokens) and tokens[index:end] == surface and not any(used[index:end]):
                if not term.apply:
                    continue
                matched = (term, end)
                break
        if matched:
            term, end = matched
            detected.append(
                {
                    "surface": term.surface,
                    "canonical": term.canonical,
                    "kind": term.kind,
                    "span": [index, end],
                    "applied": term.apply,
                }
            )
            for pos in range(index, end):
                used[pos] = True
            if term.apply:
                canonical_tokens.extend(tokenize(term.canonical) or [])
            else:
                canonical_tokens.extend(tokens[index:end])
            index = end
        else:
            canonical_tokens.append(tokens[index])
            index += 1
    return Normalisation(original=text, normalised=" ".join(canonical_tokens), detected=detected)


def retrieval_query(text: str) -> str:
    """Original tokens plus canonical expansions so both forms can match."""
    result = detect_and_normalise(text)
    return tokenize_join(text) + " " + result.normalised
