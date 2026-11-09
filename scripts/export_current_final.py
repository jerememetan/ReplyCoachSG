"""Generate the frozen current six-system replies; never grade or train models."""
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import sys

from replycoach.all_models import run_all_models
from replycoach.generate.providers import load_generator
from replycoach.final_test import validate_test_rows
from replycoach.io import ROOT, read_jsonl
from replycoach.lexicon_refinement import refine_local_reply
from replycoach.reply_bank import load_reply_bank
from replycoach.retrieval.v0 import V0Retriever
from replycoach.retrieval.v2 import V2Retriever


def main():
    out=Path(sys.argv[1]).resolve()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Current cluster generation requires an allocated CUDA GPU')
    torch.set_num_threads(4)
    assignment=json.loads((ROOT/'data/split/manifest.json').read_text(encoding='utf-8'))['assignment']
    expected={i for i,split in assignment.items() if split=='test'}
    if len(expected)!=60: raise ValueError('Expected 60 fixed test cases')
    items=validate_test_rows(read_jsonl(ROOT/'data/test/test.jsonl'),expected,
        read_jsonl(ROOT/'data/dev/train.jsonl')+read_jsonl(ROOT/'data/dev/validation.jsonl'))
    bank=load_reply_bank(ROOT/'data/dev/train.jsonl')
    if len(bank.items)!=190:raise ValueError('Expected the fixed 190-example reply bank')
    journal=out/'outputs.jsonl'
    previous={r['item_id']:r['output'] for r in read_jsonl(journal)} if journal.exists() else {}
    if not set(previous)<=expected:raise ValueError('Unexpected saved test outputs')
    from sentence_transformers import SentenceTransformer
    from replycoach.qwen_reranker import QwenReranker
    bm25=V0Retriever.from_items(bank.items)
    encoder=SentenceTransformer(str(ROOT/'artifacts/minilm_retrieval'),device='cuda')
    minilm=V2Retriever.from_items(bank.items,model=encoder,cache_file=None)
    v4=load_generator('v4');v4.device=torch.device('cuda');v4.model.to(v4.device)
    v5=load_generator('sealion_handoff')
    scorer=QwenReranker(ROOT/'artifacts/reranker_qwen/pretrained',
        adapter=ROOT/'artifacts/reranker_qwen/dd_result_20261029/stage2_reviewed/best',device='cuda')
    with journal.open('a',encoding='utf-8') as handle:
        for item in items:
            if item.id in previous:continue
            result=run_all_models(item.message.text,item.memory_texts(),'M2',
                speakers=[t.speaker for t in item.session_memory],log=False,
                bm25_factory=lambda:bm25, minilm_factory=lambda:minilm,
                v4_factory=lambda:v4,v5_factory=lambda:v5,
                scorer_factory=lambda:scorer)
            if not result.complete:
                raise RuntimeError('Incomplete comparison: '+str(result.source_errors)+' '+str(result.ranking_error))
            refinement=refine_local_reply(result.ranking.local,scorer,result.memory_speakers,
                                          rng=random.Random('425:'+item.id))
            if not refinement.complete:raise RuntimeError('V7 failed: '+str(refinement.error))
            result=replace(result,local_refinement=refinement)
            if any(c.meta.get('generation_source')!='SEALION' for c in result.proposals['V5']):
                raise ValueError('V5 did not use the selected SEA-LION model')
            row=dict(item_id=item.id,evaluation_split='test',output=result.to_dict())
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            previous[item.id]=row['output']
            progress=dict(phase='generating',completed=len(previous),total=60,
                          gpu=torch.cuda.get_device_name(0),v5='SEA-LION 3B')
            temp=out/'progress.tmp';temp.write_text(json.dumps(progress),encoding='utf-8');temp.replace(out/'progress.json')
            print(json.dumps(progress),flush=True)


if __name__=='__main__':main()
