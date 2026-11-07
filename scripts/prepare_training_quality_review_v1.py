"""Prepare a small human review packet without changing training data."""
import copy,csv,hashlib,json,random
from pathlib import Path
from human_expansion_review import prepare
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/human_review/training_quality_v1'
def load(p):return [json.loads(l) for l in (ROOT/p).read_text().splitlines() if l.strip()]
def signature(r):return json.dumps([r['session_memory'],r['message']['text']],sort_keys=True).casefold()
def main():
    if BASE.exists():raise FileExistsError('Preserve existing reviews; use a new version')
    source=ROOT/'data/training_expansion/blanket600_v1/expanded.jsonl';before=hashlib.sha256(source.read_bytes()).hexdigest()
    expanded=load(source);original={r['id']:r for r in load('data/dev/train.jsonl')};val=load('data/dev/validation.jsonl')
    rng=random.Random(1120)
    groups={'original_targets_revised':[r for r in expanded if r['id'] in original], 'generated_fresh100':[r for r in expanded if r['id'].startswith('SYN-HUMAN-EXP1-')], 'generated_volume500':[r for r in expanded if r['id'].startswith('SYN-volume500-')]}
    if sorted(map(len,groups.values()))!=[100,190,500]:raise ValueError('Unexpected corpus lineage')
    selected=[]
    for group,items in groups.items():
        for row in rng.sample(items,10):
            r=copy.deepcopy(row);r['scenario_group']=group
            r['provenance']={**r.get('provenance',{}),'quality_review_source_sha256':before,'quality_review_scope':'New explicit review of exact expanded-dataset targets; prior blanket authorization is not a new per-item rating.'}
            selected.append(r)
    if len({r['id'] for r in selected})!=30 or {signature(r) for r in selected}&{signature(r) for r in val}:raise ValueError('Duplicate or validation input overlap')
    BASE.mkdir(parents=True);candidates=BASE/'candidates.jsonl';candidates.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    packet=BASE/'review';prepare(candidates,packet,batch_size=10)
    with (BASE/'original_target_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','original_standard','expanded_standard','original_local','expanded_local'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in selected:
            if r['id'] in original:
                old=original[r['id']]['replies'];new=r['replies'];writer.writerow(dict(id=r['id'],original_standard=old['standard'],expanded_standard=new['standard'],original_local=old['local'],expanded_local=new['local']))
    (BASE/'selection_manifest.json').write_text(json.dumps(dict(seed=1120,source=str(source.relative_to(ROOT)),source_sha256=before,total=30,counts={k:10 for k in groups},purpose='Training target quality review, not model evaluation. No original locked test read. Original validation excluded. Original drafts and dataset unchanged.',human_reviews_submitted=0,assistant_suggestions_are_human_approvals=False),indent=2)+'\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    print('Prepared30 training-target pairs in3 review batches. Source unchanged;0 new human approvals.')
if __name__=='__main__':main()
