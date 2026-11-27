"""Collect fixed-English rewrite comparisons; no automatic style winner."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
ARMS=['old_targets_control','revised_targets']


def read(p):return json.loads((ROOT/p).read_text())
def save(p,v):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def comparison(reports,name):
    first=next(iter(reports.values()));ids=[r['item_id'] for r in first['outputs']]
    assert all([r['item_id'] for r in report['outputs']]==ids for report in reports.values())
    rows=[]
    for i,row in enumerate(first['outputs']):
        assert all(report['outputs'][i]['english']==row['english'] for report in reports.values())
        rows.append({**{k:row.get(k) for k in ['item_id','scope','context','speakers','message','english','reference','anchor_provenance']},
            'outputs':{a:dict(singlish=report['outputs'][i]['singlish'],status=report['outputs'][i]['status']) for a,report in reports.items()}})
    save('review/'+name+'.json',rows)
    fields=['item_id','scope','context','speakers','message','english','reference']+list(reports)
    with (ROOT/'review'/f'{name}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({**{k:json.dumps(row[k]) if k in {'context','speakers'} else row[k] for k in fields[:7]},
            **{a:out['singlish'] for a,out in row['outputs'].items()}})


def main():
    recipe=read('recipe.json');reports={};issues=[]
    for arm in ARMS:
        p=ROOT/'outputs'/arm/'training.json'
        if not p.exists():issues.append(arm+': missing report');continue
        reports[arm]=read(p)
        if reports[arm]['status']!='complete':issues.append(arm+': '+reports[arm]['status'])
    summary=dict(status='needs_attention' if issues else 'complete',issues=issues,recipe=recipe,launch=read('launch.json'),training=reports,
        default_changed=False,english_generator_changed=False,human_style_review_pending=True,
        limits='One-seed target-wording pilot with assistant-authored style targets.20shared casual pairs may help both arms. Loss/copy/particle counts are not naturalness or meaning scores. Fresh test examples are assistant-authored, old validation/demos reused, no independent human accuracy.')
    if not issues:
        assert len({r['initial_digest'] for r in reports.values()})==1
        assert len({r['schedule_sha256'] for r in reports.values()})==1
        assert all(r['optimizer_steps']==240 and r['frozen_base_unchanged'] and r['source_english_and_singlish_unchanged'] for r in reports.values())
        original=read('outputs/original/evaluation.json')
        by_id={r['item_id']:r for r in original['outputs']}
        selected_ids=[r['item_id'] for r in read('outputs/'+ARMS[0]+'/selected_evaluation.json')['outputs']]
        baseline=dict(original,outputs=[by_id[i] for i in selected_ids])
        for label in ['selected','final']:
            ev={'original':baseline}
            for arm in ARMS:
                use='selected' if label=='final' and reports[arm].get('final_reuses_selected') else label
                ev[arm]=read('outputs/'+arm+'/'+use+'_evaluation.json')
            comparison(ev,label+'_test_comparison')
            summary[label]={a:dict(flags=r['flags'],exact_copies=sum(x['english'].strip().casefold()==x['singlish'].strip().casefold() for x in r['outputs'])) for a,r in ev.items()}
        val_ids=[r['item_id'] for r in read('data/validation.json')]
        val={'original':dict(original,outputs=[by_id[i] for i in val_ids])}
        for arm in ARMS:
            for step in recipe['snapshots']:
                val[arm+'_'+str(step)]=read(f'outputs/{arm}/validation_{step:04d}.json')
        comparison(val,'validation_snapshots')
        inventory=[]
        for arm in ARMS:
            for h in reports[arm]['history']:
                p=ROOT/'outputs'/arm/f"step_{h['step']:04d}/adapter_model.safetensors"
                inventory.append(dict(arm=arm,step=h['step'],path=str(p),sha256=sha(p)))
        save('checkpoint_inventory.json',inventory)
        summary['selected_steps']={a:r['selected_step'] for a,r in reports.items()}
        summary['fixed_english_comparison_verified']=True
    save('summary.json',summary)
    files=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if p.is_symlink() or not p.is_file() or '__pycache__' in rel.parts or any(x.startswith('step_') or x=='selected' for x in rel.parts):continue
        if p.name.startswith('results_archive'):continue
        if p.suffix in {'.json','.jsonl','.py','.csv','.md','.out','.sbatch'}:files.append(p)
    archive=ROOT/'results_only_v1.tgz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    record=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),weights_included=False)
    save('results_archive.json',record)
    print(json.dumps(dict(status=summary['status'],issues=issues,archive=record)),flush=True)


if __name__=='__main__':main()
