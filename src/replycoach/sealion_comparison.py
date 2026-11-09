"""Demo comparison with SEA-LION replies passed through without reranking."""
from dataclasses import dataclass,replace
from time import perf_counter

from replycoach.all_models import SOURCE_NAMES,align_memory_speakers
from replycoach.logging_utils import log_event
from replycoach.pipeline import PreparedInput,prepare_input,rank_result
from replycoach.schema import Candidate
from replycoach.shared_ranker import build_query
from replycoach.systems import _generate,_retrieve
from replycoach.lexical_refinement_v2 import LexicalRefinement,refine_lexical_local


@dataclass
class SeaLionComparison:
    prepared: PreparedInput
    proposals: dict[str,list[Candidate]]
    source_errors: dict[str,str]
    source_seconds: dict[str,float]
    memory_speakers: list[str]
    local_refinement: LexicalRefinement|None=None

    @property
    def available_sources(self):
        return [name for name in SOURCE_NAMES if name in self.proposals]

    def _reply(self,style):
        return next((c for c in self.proposals.get('V5',[]) if c.style==style),None)

    @property
    def standard(self):return self._reply('STANDARD')

    @property
    def local(self):
        return self.local_refinement.selected if self.local_refinement is not None else self._reply('LOCAL')

    @property
    def complete(self):
        return (len(self.proposals)==4 and not self.source_errors and self.standard is not None and self.local is not None
                and (self.local_refinement is None or self.local_refinement.complete))

    def to_dict(self):
        refinement=self.local_refinement
        return dict(record_type='sealion_local_lexical_comparison' if refinement else 'sealion_direct_comparison',complete=self.complete,
            requested_sources=list(SOURCE_NAMES),available_sources=self.available_sources,
            selection_method='sealion_standard_lexical_v7_local' if refinement else 'sealion_direct',final_source='V5',
            reranker=(refinement.ranking.candidates[0].meta['ranker'] if refinement and refinement.ranking else None),
            v7=refinement.to_dict() if refinement else None,
            message=self.prepared.message,memory=self.prepared.memory,memory_speakers=self.memory_speakers,
            source_errors=dict(self.source_errors),source_seconds=dict(self.source_seconds),
            proposals={name:[c.to_dict() for c in cs] for name,cs in self.proposals.items()},
            standard=self.standard.to_dict() if self.standard else None,
            local=self.local.to_dict() if self.local else None)


def run_sealion_comparison(message,session,memory_mode='M2',*,bm25_factory=None,
                           minilm_factory=None,v4_factory=None,v5_factory=None,
                           speakers=None,log=False):
    build_query(message,session,speakers)
    prepared=prepare_input(message,session,memory_mode)
    memory_speakers=align_memory_speakers(session,speakers,prepared.memory)
    proposals,errors,times={},{},{}
    for name,factory,method in [('BM25',bm25_factory,'bm25'),('MiniLM',minilm_factory,'minilm'),
                                ('V4',v4_factory,None),('V5',v5_factory,None)]:
        started=perf_counter()
        try:
            candidates=_retrieve(prepared,factory,method,3) if method else _generate(prepared,factory)
            if name=='V5':
                if (len(candidates)!=2 or {c.style for c in candidates}!={'STANDARD','LOCAL'}
                        or any(c.source!='SEALION' for c in candidates)):
                    raise ValueError('SEA-LION must supply exactly one original reply per style')
                # Do not apply rank_result's heuristic/fallback policy to final replies.
                proposals[name]=[replace(c,context=list(prepared.memory),message=prepared.message,
                    normalised_message=prepared.normalised_message,
                    meta={**c.meta,'candidate_source':c.source,'generation_source':'SEALION',
                          'selection_method':'sealion_direct'}) for c in candidates]
            else:
                result=rank_result(prepared,candidates,'retrieval_only' if method else 'generator_only',
                                   retrieval_method=method)
                proposals[name]=[result.standard,result.local]
        except Exception as exc:
            errors[name]=str(exc) or type(exc).__name__
        times[name]=perf_counter()-started
    output=SeaLionComparison(prepared,proposals,errors,times,memory_speakers)
    if log:log_event('sealion_direct_suggestions',output.to_dict())
    return output


def run_sealion_local_refinement(message,session,memory_mode='M2',*,scorer_factory,rng=None,log=False,**sources):
    """Keep Standard direct; refine only SEA-LION Local, never the other models."""
    output=run_sealion_comparison(message,session,memory_mode,log=False,**sources)
    if output.local is not None:
        output=replace(output,local_refinement=refine_lexical_local(
            output.local,scorer_factory,output.memory_speakers,rng=rng))
    if log:log_event('sealion_lexical_suggestions',output.to_dict())
    return output
