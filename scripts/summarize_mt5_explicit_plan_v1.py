"""Package explicit-planning comparisons without mixing oracle and real routes."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads((ROOT/p).read_text())
def save(p,value):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    reports={};issues=[]
    for stage in ['planner','no_notes_control','explicit_notes']:
        p=ROOT/'outputs'/stage/'training.json'
        if not p.exists():issues.append(stage+': missing report');continue
        r=read(p);reports[stage]=r
        if r['status']!='complete':issues.append(stage+': '+r['status'])
    summary=dict(status='needs_attention' if issues else 'complete',issues=issues,recipe=read('recipe.json'),launch=read('launch.json'),
        training=reports,default_changed=False,human_review_pending=True,locked_test_read=False,
        limits='Predicted notes are actual learned-model outputs; oracle notes are labelled diagnostics, not real inference or guaranteed upper bounds. One seed, assistant labels, reused diagnostics and parameter-held-out families; no independent human accuracy.')
    if not issues:
        assert reports['no_notes_control']['initial_digest']==reports['explicit_notes']['initial_digest']
        assert reports['no_notes_control']['schedule_sha256']==reports['explicit_notes']['schedule_sha256']
        assert reports['no_notes_control']['cache_sha256']==reports['explicit_notes']['cache_sha256']
        assert all(reports[a]['optimizer_steps']==300 and reports[a]['source_base_and_initial_adapter_unchanged'] for a in ['no_notes_control','explicit_notes'])
        baseline=read('outputs/unchanged_baseline/reference_evaluation.json')['routes']['standard']
        ids=[r['item_id'] for r in baseline]
        for label in ['selected','final']:
            ev={}
            for a in ['no_notes_control','explicit_notes']:
                use='selected' if label=='final' and reports[a].get('final_reuses_selected') else label
                ev[a]=read('outputs/'+a+'/'+use+'_evaluation.json')
            routes={'unchanged_baseline':baseline,'continued_control':ev['no_notes_control']['routes']['no_notes_control'],
                'predicted_notes':ev['explicit_notes']['routes']['predicted_notes'],'same_model_no_notes':ev['explicit_notes']['routes']['same_model_no_notes'],
                'oracle_notes_DIAGNOSTIC':ev['explicit_notes']['routes']['oracle_notes_diagnostic']}
            assert all([r['item_id'] for r in routes[a]]==ids for a in ['continued_control','predicted_notes','same_model_no_notes'])
            by_route={a:{r['item_id']:r for r in rs} for a,rs in routes.items()}
            cache=read('predicted_notes.json')
            for row in routes['predicted_notes']:
                assert row['notes']==cache[row['item_id']]['plan']
            rows=[]
            for r in baseline:
                rows.append({**{k:r[k] for k in ['item_id','scope','context','speakers','message']},
                    'predicted_notes':cache[r['item_id']],
                    'routes':{a:rs.get(r['item_id']) for a,rs in by_route.items()}})
            save('review/'+label+'_comparison.json',rows)
            fields=['item_id','scope','context','speakers','message','predicted_notes']+[a+'_'+s for a in routes for s in ['english','singlish']]
            with (ROOT/'review'/f'{label}_comparison.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for row in rows:
                    writer.writerow({**{k:json.dumps(row[k]) if k in {'context','speakers','predicted_notes'} else row[k] for k in fields[:6]},
                        **{a+'_'+s:(out[s] if out else '') for a,out in row['routes'].items() for s in ['english','singlish']}})
            summary[label]=dict(control_step=reports['no_notes_control']['selected_step'] if label=='selected' else 300,
                explicit_step=reports['explicit_notes']['selected_step'] if label=='selected' else 300,
                cases=len(ids),oracle_cases=len(routes['oracle_notes_DIAGNOSTIC']),
                predictions_changed_by_notes=sum(a['english']!=b['english'] for a,b in zip(routes['predicted_notes'],routes['same_model_no_notes'],strict=True)),
                note='String changes do not establish helpful use of notes.',flags={a:r['flags'] for a,r in ev.items()})
        summary['predicted_notes_passed_exactly_verified']=True
    save('summary.json',summary)
    files=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if p.is_symlink() or not p.is_file() or '__pycache__' in rel.parts or any(x.startswith('step_') or x=='selected' for x in rel.parts):continue
        if p.name.startswith('results_archive'):continue
        if p.suffix in {'.json','.jsonl','.py','.csv','.out','.sbatch','.md'}:files.append(p)
    archive=ROOT/'results_only_v1.tgz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    record=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),weights_included=False)
    save('results_archive.json',record)
    print(json.dumps(dict(status=summary['status'],issues=issues,archive=record)),flush=True)


if __name__=='__main__':main()
