"""Archive the bounded position-balanced follow-up and its full comparisons."""
import json
import os
from pathlib import Path
import tarfile
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import summarize_sealion_abstention_v1 as shared


def main():
    assert Path.home().name=='hm.teoh.2024'
    read,save,sha=shared.read,shared.save,shared.sha
    study=read('results.json');assert study['status']=='complete' and study['source_adapters_unchanged']
    arm=study['arms']['selective_abstention'];step=arm['selected_step']
    assert arm['optimizer_steps']==280 and arm['frozen_base_unchanged']
    reports={'original':read('outputs/baseline/evaluation.json'),'abstention_v1_step140':read('outputs/v1_reference/evaluation.json'),
        'position_v2_selected_step'+str(step):read('outputs/selective_abstention/selected/evaluation.json')}
    reports['position_v2_final280']=reports['position_v2_selected_step'+str(step)] if step==280 else read('outputs/selective_abstention/final280/evaluation.json')
    rows=shared.compare(reports,'fresh_synthetic_comparison')
    custom={name:read('custom_check/'+name+'.json') for name in ['original','v1','v2']}
    assert all(r['status']=='complete' and not r['context_rule_layer_used'] for r in custom.values())
    custom_rows=shared.compare(custom,'custom_diagnostics')
    inventory=[]
    for name in ['selected','step_0070','step_0140','step_0280']:
        p=ROOT/'outputs/selective_abstention'/name
        inventory.append(dict(label=name,path=str(p),bytes=(p/'adapter_model.safetensors').stat().st_size,
            sha256=sha(p/'adapter_model.safetensors'),config_sha256=sha(p/'adapter_config.json')))
    save('checkpoint_inventory.json',inventory)
    if step==0:
        assert [(r['english'],r['singlish']) for r in reports['position_v2_selected_step0']['outputs']]==[(r['english'],r['singlish']) for r in reports['original']['outputs']]
    quota_limit=int(os.getxattr(Path.home(),'ceph.quota.max_bytes'))
    quota_used=int(os.getxattr(Path.home(),'ceph.dir.rbytes'))
    summary=dict(status='all_experiments_complete',account='hm.teoh.2024',root=str(ROOT),
        jobs=dict(training=358104,custom_inference_comparison=358112),
        recipe=study['recipe'],audit=read('audit.json'),preflight=read('preflight.json'),training=study,
        main_metrics={name:report['metrics'] for name,report in reports.items()},
        custom_diagnostics=custom_rows,checkpoint_count=4,new_trained_snapshots=3,
        source_adapters_unchanged=True,default_changed=False,locked_test_read=False,human_review_pending=True,
        selected_step=step,selected_sha256=arm['selected_sha256'],
        quota_bytes=dict(limit=quota_limit,used=quota_used,free=quota_limit-quota_used),
        main_comparison='review/fresh_synthetic_comparison.csv',custom_comparison='review/custom_diagnostics.csv',
        data_authoring_issue='Subsequent audit found Friend questions used I/my rather than you/your for six User-fact families, potentially invalidating72 supported augmentation rows. V1/V2 are retained as diagnostic experiments, not approved models. V3 corrects these references, adds tests and resets from original weights.',
        limits='Twenty fresh synthetic inputs comprise8 paired fact scenarios plus4 self-contained controls; not20 independent human conversations. The96 validation rows are two views of48 scenarios. Exact standard-refusal and lexical match diagnostics are not semantic accuracy. V1-derived diagnostics informed this data revision; no newly matched restored-context control for V2. No automatic promotion.')
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
    print(json.dumps(dict(status=summary['status'],archive=record,selected_step=step,
        main_metrics=summary['main_metrics'],custom=custom_rows)),flush=True)


if __name__=='__main__':main()
