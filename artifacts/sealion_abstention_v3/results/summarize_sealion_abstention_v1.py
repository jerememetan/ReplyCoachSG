"""Collect the completed abstention experiment without downloading any weights."""
import csv
import hashlib
import json
import os
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent


def read(name):return json.loads((ROOT/name).read_text())


def save(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def compare(reports,name):
    first=next(iter(reports.values()))
    ids=[r['item_id'] for r in first['outputs']]
    assert all(r['status']=='complete' and [x['item_id'] for x in r['outputs']]==ids for r in reports.values())
    rows=[]
    for i,ident in enumerate(ids):
        original=first['outputs'][i]
        rows.append(dict(item_id=ident,scope=original['scope'],kind=original.get('kind'),
            requires_refusal=original['requires_refusal'],context=list(zip(original['speakers'],original['context'],strict=True)),
            message=original['message'],reference=original.get('positive'),
            outputs={key:{field:report['outputs'][i].get(field) for field in ['english','singlish','canonical_abstention','needs_more_context','english_status','singlish_status']}
                for key,report in reports.items()}))
    save(f'review/{name}.json',rows)
    fields=['item_id','scope','kind','requires_refusal','context','message','reference']+[key+'_'+style for key in reports for style in ['english','singlish']]
    with (ROOT/f'review/{name}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row[k] for k in fields[:7] if k!='context'},
                'context':json.dumps(row['context'],ensure_ascii=False),
                **{key+'_'+style:out[style] for key,out in row['outputs'].items() for style in ['english','singlish']}})
    return rows


def main():
    assert Path.home().name=='hm.teoh.2024'
    training=read('results.json');probe=read('probe/results.json')
    amendment=read('validation_amendment/results.json')
    custom=read('custom_check/results.json')
    assert training['status']==probe['status']=='complete'
    assert training['source_adapters_unchanged'] and training['matched_initial_weights_and_slot_schedules']
    assert all(not row['selected_step_changed'] for row in amendment['arms'].values())
    assert custom['status']=='complete' and not custom['context_rule_layer_used']
    baseline=read('outputs/baseline/evaluation.json')
    reports={'original':baseline}
    inventory=[]
    selected_replays={}
    for arm,short in [('restored_context_control','control'),('selective_abstention','augmented')]:
        assert training['arms'][arm]['optimizer_steps']==280 and training['arms'][arm]['frozen_base_unchanged']
        step=training['arms'][arm]['selected_step']
        selected=read(f'outputs/{arm}/selected/evaluation.json')
        reports[f'{short}_selected_step{step}']=selected
        reports[short+'_final280']=selected if step==280 else read(f'outputs/{arm}/final280/evaluation.json')
        if step==0:
            differences=[a['item_id'] for a,b in zip(selected['outputs'],baseline['outputs'],strict=True) if (a['english'],a['singlish'])!=(b['english'],b['singlish'])]
            assert not differences
            selected_replays[arm]=dict(step=0,differences=differences)
        for label in ['selected','step_0070','step_0140','step_0280']:
            path=ROOT/'outputs'/arm/label
            inventory.append(dict(arm=arm,label=label,selected_step=step,path=str(path),
                bytes=(path/'adapter_model.safetensors').stat().st_size,
                sha256=sha(path/'adapter_model.safetensors'),config_sha256=sha(path/'adapter_config.json')))
    rows=compare(reports,'main_comparison')
    probe_reports={name:read('probe/outputs/'+name+'.json') for name in probe['reports']}
    compare(probe_reports,'presentation_probe_comparison')
    save('checkpoint_inventory.json',inventory)
    quota_limit=int(os.getxattr(Path.home(),'ceph.quota.max_bytes'))
    quota_used=int(os.getxattr(Path.home(),'ceph.dir.rbytes'))
    summary=dict(status='all_experiments_complete',account='hm.teoh.2024',
        jobs=dict(training=358095,presentation_probe=358096,custom_replay=358102),root=str(ROOT),recipe=training['recipe'],audit=read('audit.json'),
        preflight=read('preflight.json'),training=training,
        main_metrics={name:report['metrics'] for name,report in reports.items()},probe=probe,
        validation_label_amendment=amendment,custom_inference=custom,
        followup='Position-balanced v2 planned after the new adapter refused a call even with its purpose supplied earlier in the history. The original augmentation usually placed supported facts last; do not claim general robustness from canonical-refusal counts alone.',
        selected_step0_replays=selected_replays,checkpoint_count=len(inventory),new_trained_snapshots=6,
        source_adapters_unchanged=True,default_changed=False,human_review_pending=True,locked_test_read=False,
        user_case=next(r for r in rows if r['item_id']=='USER-CALL-001'),
        quota_bytes=dict(limit=quota_limit,used=quota_used,free=quota_limit-quota_used),
        main_comparison='review/main_comparison.csv',probe_comparison='review/presentation_probe_comparison.csv',
        limits='Exact standard-refusal decisions and lexical value checks are not semantic correctness. Baseline noncanonical uncertainty is counted separately. Synthetic validation/challenge; probe reuses the same challenge facts under changed presentation. No independent human or locked-project-test result.')
    save('summary.json',summary)
    archive=ROOT/'results_only_v1.tgz';files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(ROOT)
        if '__pycache__' in rel.parts or any(part=='selected' or part.startswith('step_') for part in rel.parts[:-1]):continue
        if p.name.startswith('results_archive'):continue
        if p.suffix in {'.json','.py','.csv','.sbatch','.out'}:files.append(p)
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    record=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),weights_included=False)
    save('results_archive.json',record)
    print(json.dumps(dict(status=summary['status'],archive=record,main_metrics=summary['main_metrics'],
        probe=probe['reports'],user_case=summary['user_case'])),flush=True)


if __name__=='__main__':main()
