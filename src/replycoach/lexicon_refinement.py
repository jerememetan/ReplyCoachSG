"""V7: three local lexicon alternatives, then the unchanged four-pair scorer."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import random
import re
from time import perf_counter

from replycoach.lexicon import LocalTerm, load_lexicon
from replycoach.schema import Candidate
from replycoach.shared_ranker import PairScorer, SharedRanking, rank_proposals

_LEXICAL_KINDS = {'lexicon', 'request', 'stance', 'negation', 'time'}
# Other particles (especially "meh") have stronger pragmatic implications.
_ENDINGS = {'lah', 'leh', 'lor', 'sia'}
_PROTECTED = re.compile(r'https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[\w-]+|`[^`]*`', re.I)
_PUNCTUATION = re.compile(r'''[.!?…。！？,;:，；："'”’)\]]*$''')


def _key(text: str) -> str:
    return ' '.join(text.split()).casefold()


def generate_local_variants(reply: str, *, lexicon: list[LocalTerm] | None = None,
                            rng: random.Random | None = None) -> list[str]:
    """Sample exactly three unique alternatives; never invoke a generator/model.

    Each pool entry uses one lexical substitution, one ending change, or both.
    The pool grows linearly with matches, with no retry loop or cascading edits.
    ``rng`` and ``lexicon`` allow reproducible offline checks, not UI settings.
    """
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError('reply must be a nonempty string')
    terms = load_lexicon() if lexicon is None else lexicon
    protected = [match.span() for match in _PROTECTED.finditer(reply)]
    lexical = []
    for term in sorted(terms, key=lambda t: (-len(t.canonical.split()), t.surface)):
        canonical, surface = term.canonical.strip(), term.surface.strip()
        if (not term.apply or term.kind not in _LEXICAL_KINDS or not canonical or not surface
                or not surface.isascii() or not canonical.isascii() or _key(surface) == _key(canonical)):
            continue
        pattern = r'(?<!\w)' + r'\s+'.join(re.escape(word) for word in canonical.split()) + r'(?!\w)'
        for match in re.finditer(pattern, reply, re.I):
            if any(match.start() < end and match.end() > start for start, end in protected):
                continue
            local = surface.upper() if match.group().isupper() else surface
            if not match.group().isupper() and match.group()[0].isupper():
                local = local[0].upper() + local[1:]
            lexical.append(reply[:match.start()] + local + reply[match.end():])

    particles = sorted({t.surface.strip().lower() for t in terms
                        if t.apply and t.kind == 'particle' and t.surface.strip()})
    endings = [particle for particle in particles if particle in _ENDINGS]
    # Require standalone endings; a URL path or email suffix may also say "lah".
    terminal = (re.compile(r'(?<!\S)(?:(?:' + '|'.join(map(re.escape, particles)) + r')\s*)+$', re.I)
                if particles else None)
    pool = {}

    def add(text):
        normalized = _key(text)
        if normalized != _key(reply):
            pool.setdefault(normalized, text)

    for changed in lexical:
        add(changed)
    for text in [reply, *lexical]:
        text = text.rstrip()
        ending = _PUNCTUATION.search(text)
        body, punctuation = text[:ending.start()].rstrip(), ending.group()
        if terminal:
            body = terminal.sub('', body).rstrip()
        for particle in endings:
            add(f'{body} {particle}{punctuation}'.lstrip())
    if len(pool) < 3:
        raise ValueError('lexicon cannot produce three distinct local variations')
    return (rng if rng is not None else random.Random()).sample(list(pool.values()), 3)


@dataclass
class LocalRefinement:
    original: Candidate
    variations: list[str] = field(default_factory=list)
    ranking: SharedRanking | None = None
    generation_seconds: float = 0.
    ranking_seconds: float = 0.
    error: str | None = None

    @property
    def complete(self) -> bool:
        return (self.error is None and len(self.variations) == 3 and self.ranking is not None
                and len(self.ranking.candidates) == 4 and self.ranking.local is not None)

    @property
    def selected(self) -> Candidate:
        return self.ranking.local if self.complete else self.original

    def to_dict(self) -> dict:
        return {'record_type': 'v7_local_refinement', 'complete': self.complete,
                'original': self.original.to_dict(), 'variations': list(self.variations),
                'candidate_count': 1 + len(self.variations),
                'ranked_candidates': [c.to_dict() for c in self.ranking.candidates] if self.ranking else [],
                'selected': self.selected.to_dict(), 'generation_seconds': self.generation_seconds,
                'ranking_seconds': self.ranking_seconds, 'error': self.error}


def refine_local_reply(original: Candidate, scorer: PairScorer,
                       speakers: list[str] | None = None, *,
                       rng: random.Random | None = None) -> LocalRefinement:
    """Refine only the first-pass LOCAL winner; preserve it on visible failure."""
    original.validate()
    if original.style != 'LOCAL':
        raise ValueError('V7 requires the selected LOCAL reply')
    result = LocalRefinement(original)
    started = perf_counter()
    try:
        if rng is None:
            result.variations = generate_local_variants(original.candidate_text)
        else:
            result.variations = generate_local_variants(original.candidate_text, rng=rng)
    except Exception as exc:
        result.error = str(exc) or type(exc).__name__
    result.generation_seconds = perf_counter() - started
    if result.error:
        return result
    started = perf_counter()
    try:
        texts = [original.candidate_text, *result.variations]
        ranking = rank_proposals(original.message, original.context,
                                 {'V7': [replace(original, candidate_text=text) for text in texts]},
                                 scorer, speakers)
        refined = []
        for candidate in ranking.candidates:
            index = next(i for i, text in enumerate(texts) if _key(text) == _key(candidate.candidate_text))
            meta = {**original.meta, **candidate.meta,
                    'contributing_models': list(original.meta.get('contributing_models', [])),
                    'fallback_models': list(original.meta.get('fallback_models', [])),
                    'original_proposals': list(original.meta.get('original_proposals', [])),
                    'v7': {'method': 'lexicon_variations_v1', 'original_reply': original.candidate_text,
                           'original_score': original.score, 'variant_index': index, 'is_original': index == 0}}
            refined.append(replace(candidate, source='V7', meta=meta,
                                   item_id=original.item_id if index == 0 else None))
        result.ranking = SharedRanking(refined, None, refined[0])
    except Exception as exc:
        result.error = str(exc) or type(exc).__name__
    result.ranking_seconds = perf_counter() - started
    return result
