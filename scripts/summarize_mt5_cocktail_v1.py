"""Gather the four cocktail arms, including failed-job evidence, without weights."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ARMS=['baseline','facts','answerability','both']


def read(p):return json.loads((ROOT/p).read_text())
def save(p,value):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    recipe=read('recipe.json');reports={};issues=[]
    for arm in ARMS:
        p=ROOT/'outputs'/arm/'report.json'
        if not p.exists():issues.append(arm+': missing report');continue
        report=read(p);reports[arm]=report
        if report['status']!='complete':issues.append(arm+': '+report['status'])
    summary=dict(status='needs_attention' if issues else 'complete',issues=issues,
        recipe=recipe,launch=read('launch.json'),reports=reports,
        default_changed=False,human_review_pending=True,locked_test_read=False,
        limits='One-seed LoRA pilot with generated auxiliary labels, reused project validation and assistant-authored prospective challenges. Exact auxiliary labels are not semantic reply quality. No independent human or locked-test accuracy, full-finetuning replication or general-instruction dataset sweep.')
    if not issues:
        assert len({r['initial_adapter_digest'] for r in reports.values()})==1
        assert len({r['schedule_sha256'] for r in reports.values()})==1
        assert all(r['optimizer_steps']==784 and r['primary_exposures']==9408 and r['frozen_base_unchanged'] and r['source_base_manifest_unchanged'] for r in reports.values())
        source_rows=sum([read('data/'+name+'.json') for name in ['challenge','natural_challenge','legacy','diagnostics']],[])
        summary['selected']={};summary['final']={}
        for selection in ['selected','final']:
            evaluations={arm:read('outputs/'+arm+'/'+('selected' if selection=='final' and reports[arm].get('final_reuses_selected') else selection)+'_evaluation.json') for arm in ARMS}
            assert all([r['item_id'] for r in e['outputs']]==[r['item_id'] for r in source_rows] for e in evaluations.values())
            rows=[]
            for i,source in enumerate(source_rows):
                row={k:source.get(k) for k in ['item_id','scope','context','speakers','message','review_constraints']}
                row['outputs']={arm:e['outputs'][i] for arm,e in evaluations.items()};rows.append(row)
            save('review/'+selection+'_comparison.json',rows)
            fields=['item_id','scope','context','speakers','message','review_constraints']+[a+'_'+s for a in ARMS for s in ['english','singlish']]
            with (ROOT/'review'/f'{selection}_comparison.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for row in rows:
                    writer.writerow({**{k:json.dumps(row[k]) if k in {'context','speakers'} else row[k] for k in fields[:6]},
                        **{a+'_'+s:row['outputs'][a][s] for a in ARMS for s in ['english','singlish']}})
            summary[selection]={arm:dict(step=reports[arm]['selected_step'] if selection=='selected' else 784,
                auxiliary_challenge=e['auxiliary_challenge']['metrics'],auxiliary_validation=e['auxiliary_validation']['metrics'],
                generation_flags=e['flags']) for arm,e in evaluations.items()}
        summary['matching_verified']=True
    save('summary.json',summary)
    files=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if p.is_symlink() or not p.is_file() or '__pycache__' in rel.parts or any(part.startswith('step_') for part in rel.parts):continue
        if p.name.startswith('results_archive'):continue
        if p.suffix in {'.json','.py','.csv','.md','.out','.sbatch'}:files.append(p)
    archive=ROOT/'results_only_v1.tgz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    record=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),weights_included=False)
    save('results_archive.json',record)
    print(json.dumps(dict(status=summary['status'],issues=issues,archive=record)),flush=True)


if __name__=='__main__':main()
