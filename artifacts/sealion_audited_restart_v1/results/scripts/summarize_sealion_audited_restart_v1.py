"""Collect audited-restart results; preserve actual anchors and failures."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def save(p,v):p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    stages={};issues=[]
    for stage in ['task','rewrite']:
        path=ROOT/'outputs'/stage/'training.json'
        if not path.exists():issues.append(stage+' missing');continue
        stages[stage]=read(path)
        if stages[stage]['status']!='complete':issues.append(stage+' '+stages[stage]['status'])
    summary=dict(status='needs_attention' if issues else 'complete',issues=issues,training=stages,default_changed=False,human_review_pending=True,
        limits='Existing baseline has different training history; no isolated causal attribution to wording. Assistant-authored new data and reused development validation, not independent human accuracy.')
    if not issues:
        new=read('outputs/rewrite/selected_pipeline.json');old=read('outputs/original/baseline.json');rows=read('data/evaluation.json')
        assert len(new['outputs'])==len(old['outputs'])==len(rows)==75
        assert [r['item_id'] for r in new['outputs']]==[r['item_id'] for r in old['outputs']]==[r['item_id'] for r in rows]
        comparisons=[]
        for r,b,n in zip(rows,old['outputs'],new['outputs'],strict=True):
            assert b['english']==b['standard_anchor'] and n['english']==n['standard_anchor']
            comparisons.append(dict(item_id=r['item_id'],scope=r['scope'],context=r['context'],speakers=r['speakers'],message=r['message'],original=b,audited_restart=n))
        save('review/comparison.json',comparisons)
        with (ROOT/'review/comparison.csv').open('w',newline='') as f:
            fields=['item_id','scope','context','message','original_english','original_singlish','new_english','new_singlish'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for r in comparisons:w.writerow(dict(item_id=r['item_id'],scope=r['scope'],context=json.dumps(list(zip(r['speakers'],r['context']))),message=r['message'],original_english=r['original']['english'],original_singlish=r['original']['singlish'],new_english=r['audited_restart']['english'],new_singlish=r['audited_restart']['singlish']))
        summary.update(cases=75,new_pipeline_status=new['status'],flags=new['flags'],actual_anchors_verified=True,
            selected_epochs={k:v['selected_epoch'] for k,v in stages.items()},note='Word counts, copies and validation loss are not naturalness or meaning scores. Inspect fresh24 separately from reused51.')
    save('summary.json',summary)
    archive=ROOT/'results_only_v1.tgz';files=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if not p.is_file() or p.is_symlink() or '__pycache__' in rel.parts or any(x.startswith('epoch_') or x=='selected' for x in rel.parts):continue
        if p.suffix in {'.json','.jsonl','.csv','.py','.out','.log','.sbatch'} and not p.name.startswith('results_archive'):files.append(p)
    with tarfile.open(archive,'w:gz') as t:
        for p in files:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    record=dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),weights_included=False);save('results_archive.json',record)
    print(json.dumps(dict(status=summary['status'],issues=issues,archive=record)),flush=True)
if __name__=='__main__':main()
