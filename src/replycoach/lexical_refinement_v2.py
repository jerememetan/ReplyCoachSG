"""Independent per-phrase random shortening, followed by four-candidate scoring."""
from dataclasses import dataclass,field,replace
from functools import lru_cache
from hashlib import sha256
import json,random,re
from time import perf_counter

from replycoach.io import ROOT
from replycoach.schema import Candidate
from replycoach.shared_ranker import rank_proposals

PROBABILITY=.5
METHOD='lexical_bernoulli_v2'
_PROTECTED=re.compile(r'```[\s\S]*?```|`[^`]*`|https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[\w-]+',re.I)


@lru_cache(maxsize=1)
def _dictionary():
    raw=(ROOT/'data/lexicons/chat_variations.json').read_bytes()
    data=json.loads(raw);terms={}
    for entry in data['conversions']:
        for phrase in entry['phrases']:
            phrase=' '.join(phrase.split()).casefold()
            values=tuple(dict.fromkeys(entry['replacements']))
            if (not phrase or phrase in terms or not values or
                    any(not v.strip() or v!=v.strip() or v.casefold()==phrase for v in values)):
                raise ValueError('Invalid chat conversion dictionary')
            terms[phrase]=values
    pattern=re.compile(r'(?<!\w)(?:'+'|'.join(r'\s+'.join(map(re.escape,k.split()))
        for k in sorted(terms,key=lambda k:(-len(k),k)))+r')(?!\w)',re.I)
    return terms,pattern,sha256(raw).hexdigest()


def randomize_local_text(reply,*,rng):
    """One independent 50% roll per eligible original phrase occurrence."""
    if not isinstance(reply,str) or not reply.strip():raise ValueError('reply must be nonempty')
    terms,pattern,_=_dictionary()
    protected=[m.span() for m in _PROTECTED.finditer(reply)]
    pieces=[];position=0
    for match in pattern.finditer(reply):
        if any(match.start()<end and match.end()>start for start,end in protected):continue
        if rng.random()>=PROBABILITY:continue
        replacement=rng.choice(terms[' '.join(match.group().split()).casefold()])
        if match.group().isupper():replacement=replacement.upper()
        elif match.group()[0].isupper():replacement=replacement[0].upper()+replacement[1:]
        pieces.extend([reply[position:match.start()],replacement]);position=match.end()
    return ''.join([*pieces,reply[position:]])


def generate_lexical_variants(reply,*,rng=None):
    """Three unique changed samples, or none when a four-candidate pool cannot form.

    Rejecting unchanged/duplicate attempts conditions the final distribution;
    50% is the per-attempt edit probability, not a final-output edit quota.
    """
    rng=rng if rng is not None else random.Random()
    variants={};original=' '.join(reply.split()).casefold()
    for _ in range(64):
        text=randomize_local_text(reply,rng=rng);key=' '.join(text.split()).casefold()
        if key!=original:variants.setdefault(key,text)
        if len(variants)==3:return list(variants.values())
    return []


@dataclass
class LexicalRefinement:
    original: Candidate
    variations: list[str]=field(default_factory=list)
    ranking: object=None
    generation_seconds: float=0.
    ranking_seconds: float=0.
    skipped_reason: str|None=None
    error: str|None=None
    lexicon_sha256: str|None=None

    @property
    def complete(self):
        return self.error is None and (self.skipped_reason is not None or
            (len(self.variations)==3 and self.ranking is not None and len(self.ranking.candidates)==4
             and self.ranking.local is not None))

    @property
    def selected(self):
        return self.ranking.local if self.complete and self.ranking is not None else self.original

    def to_dict(self):
        return dict(record_type='v7_lexical_refinement',method=METHOD,replacement_probability=PROBABILITY,
            conversion_lexicon_sha256=self.lexicon_sha256,complete=self.complete,
            status='failed' if self.error else 'skipped' if self.skipped_reason else 'completed',
            skipped_reason=self.skipped_reason,error=self.error,original=self.original.to_dict(),
            variations=list(self.variations),candidate_count=1+len(self.variations),
            ranked_candidates=[c.to_dict() for c in self.ranking.candidates] if self.ranking else [],
            selected=self.selected.to_dict(),generation_seconds=self.generation_seconds,ranking_seconds=self.ranking_seconds)


def refine_lexical_local(original,scorer_factory,speakers=None,*,rng=None):
    original.validate()
    if original.style!='LOCAL' or original.source!='SEALION':
        raise ValueError('Lexical refinement requires the original SEA-LION Local reply')
    result=LexicalRefinement(original);tick=perf_counter()
    try:
        result.lexicon_sha256=_dictionary()[2]
        result.variations=generate_lexical_variants(original.candidate_text,rng=rng)
        if not result.variations:
            result.skipped_reason='Could not produce three distinct wording variations; kept the original SEA-LION reply.'
    except Exception as exc:result.error=str(exc) or type(exc).__name__
    result.generation_seconds=perf_counter()-tick
    if result.error or result.skipped_reason:return result
    tick=perf_counter()
    try:
        texts=[original.candidate_text,*result.variations]
        result.ranking=rank_proposals(original.message,original.context,
            {'V5':[replace(original,candidate_text=t) for t in texts]},scorer_factory(),speakers)
        for c in result.ranking.candidates:
            c.meta['v7']=dict(method=METHOD,is_original=c.candidate_text==original.candidate_text,
                             original_reply=original.candidate_text)
    except Exception as exc:result.error=str(exc) or type(exc).__name__
    result.ranking_seconds=perf_counter()-tick
    return result
