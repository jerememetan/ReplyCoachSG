"""Collect original/single/multiple/contrast replies and unranked candidates."""
import csv
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
import run_sealion_turn_repair_v1 as core


def main():
    recipe=core.read('recipe.json');arms=recipe['arms'];reports={};issues=[]
    for arm in arms:
        p=ROOT/'outputs'/arm/'training.json'
        if not p.exists():issues.append(arm+': missing training report');continue
        r=json.loads(p.read_text());reports[arm]=r
        if r['status']!='complete':issues.append(arm+': '+r['status'])
    summary=dict(status='needs_attention' if issues else 'complete',issues=issues,
        recipe=recipe,launch=core.read('launch.json'),training=reports,default_changed=False,
        human_review_pending=True,locked_test_read=False,
        limits='Assistant-authored references/negatives, parameter-held-out template families, reused natural/user diagnostics and one training seed. No independent human accuracy. Sampling diversity is not contextual quality. No automatic reranker or model promotion.')
    if not issues:
        assert len({r['initial_digest'] for r in reports.values()})==1
        assert len({r['schedule_sha256'] for r in reports.values()})==1
        assert all(r['optimizer_steps']==300 and r['frozen_base_unchanged'] and r['source_adapters_unchanged'] for r in reports.values())
        inventory=[]
        for arm in arms:
            for h in reports[arm]['history']:
                path=ROOT/'outputs'/arm/f"step_{h['step']:04d}"
                inventory.append(dict(arm=arm,step=h['step'],path=str(path),sha256=core.sha(path/'adapter_model.safetensors')))
        core.save('checkpoint_inventory.json',inventory)
        for selection in ['selected','final']:
            evaluations={'original':core.read('outputs/original/evaluation.json')}
            for arm in arms:
                label='selected' if selection=='final' and reports[arm].get('final_reuses_selected') else selection
                evaluations[arm]=core.read('outputs/'+arm+'/'+label+'_evaluation.json')
            original=evaluations['original'];ids=[r['item_id'] for r in original['outputs']]
            assert all(e['status']=='complete' and [r['item_id'] for r in e['outputs']]==ids for e in evaluations.values())
            rows=[]
            for i,row in enumerate(original['outputs']):
                rows.append({**{k:row.get(k) for k in ['item_id','scope','context','speakers','message','criterion']},
                    'outputs':{a:{k:e['outputs'][i][k] for k in ['english','singlish','english_status','singlish_status']} for a,e in evaluations.items()}})
            core.save('review/'+selection+'_comparison.json',rows)
            fields=['item_id','scope','context','speakers','message','criterion']+[a+'_'+s for a in evaluations for s in ['english','singlish']]
            with (ROOT/'review'/f'{selection}_comparison.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for row in rows:writer.writerow({**{k:json.dumps(row[k]) if k in {'context','speakers'} else row[k] for k in fields[:6]},**{a+'_'+s:out[s] for a,out in row['outputs'].items() for s in ['english','singlish']}})
            core.save('review/'+selection+'_sampled_candidates.json',{a:e['sampled'] for a,e in evaluations.items()})
            summary[selection]={a:dict(flags=e['flags'],sampled_status_counts=e['sampled_status_counts'],
                sampled_cases=len(e['sampled']),mean_distinct_candidates=sum(r['unique_english'] for r in e['sampled'])/len(e['sampled'])) for a,e in evaluations.items()}
        summary['selected_steps']={a:r['selected_step'] for a,r in reports.items()}
    core.save('summary.json',summary)
    files=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if p.is_symlink() or not p.is_file() or '__pycache__' in rel.parts or any(x.startswith('step_') or x=='selected' for x in rel.parts):continue
        if p.name.startswith('results_archive'):continue
        if p.suffix in {'.json','.jsonl','.py','.csv','.md','.out','.sbatch'}:files.append(p)
    archive=ROOT/'results_only_v1.tgz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    record=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=core.sha(archive),files=len(files),weights_included=False)
    core.save('results_archive.json',record)
    print(json.dumps(dict(status=summary['status'],issues=issues,archive=record)),flush=True)


if __name__=='__main__':main()
