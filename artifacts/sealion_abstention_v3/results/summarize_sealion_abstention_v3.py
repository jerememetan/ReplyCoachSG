"""Archive the corrected, bounded abstention study without model weights."""
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import summarize_sealion_abstention_v1 as shared


def main():
    assert Path.home().name == 'hm.teoh.2024'
    read, save, sha = shared.read, shared.save, shared.sha
    study = read('results.json')
    assert study['status'] == 'complete' and study['source_adapters_unchanged']
    arm = study['arms']['selective_abstention']
    step = arm['selected_step']
    assert arm['optimizer_steps'] == 288 and arm['frozen_base_unchanged']
    reports = {
        'original': read('outputs/baseline/evaluation.json'),
        f'corrected_selected_step{step}': read('outputs/selective_abstention/selected/evaluation.json'),
    }
    reports['corrected_final288'] = reports[f'corrected_selected_step{step}'] if step == 288 else read('outputs/selective_abstention/final288/evaluation.json')
    rows = shared.compare(reports, 'paired_comparison')
    custom = read('custom_replay/results.json')
    assert custom['status'] == 'complete' and not custom['context_rule_layer_used']
    expected = {r['item_id']: r for r in reports[f'corrected_selected_step{step}']['outputs']}
    assert all((r['english'], r['singlish']) == (expected[r['item_id']]['english'], expected[r['item_id']]['singlish']) for r in custom['outputs'])
    inventory = []
    for name in ['selected', 'step_0072', 'step_0144', 'step_0288']:
        p = ROOT / 'outputs/selective_abstention' / name
        inventory.append(dict(label=name, path=str(p), bytes=(p/'adapter_model.safetensors').stat().st_size,
            sha256=sha(p/'adapter_model.safetensors'), config_sha256=sha(p/'adapter_config.json')))
    save('checkpoint_inventory.json', inventory)
    if step == 0:
        assert [(r['english'], r['singlish']) for r in reports['corrected_selected_step0']['outputs']] == [(r['english'], r['singlish']) for r in reports['original']['outputs']]
    quota_limit = int(os.getxattr(Path.home(), 'ceph.quota.max_bytes'))
    quota_used = int(os.getxattr(Path.home(), 'ceph.dir.rbytes'))
    summary = dict(status='all_experiments_complete', account='hm.teoh.2024', root=str(ROOT),
        jobs=read('completed_jobs.json'), recipe=study['recipe'], audit=read('audit.json'),
        preflight=read('preflight.json'), training=study,
        main_metrics={name: report['metrics'] for name, report in reports.items()},
        selected_step=step, selected_sha256=arm['selected_sha256'],
        diagnostics=[r for r in rows if r['scope']=='reused_user_diagnostic'],
        custom_replay_matches_training_evaluation=True,
        checkpoint_count=4, new_trained_snapshots=3, source_adapters_unchanged=True,
        default_changed=False, locked_test_read=False, human_review_pending=True,
        quota_bytes=dict(limit=quota_limit, used=quota_used, free=quota_limit-quota_used),
        data_quality_correction=read('data_quality_correction.json'),
        main_comparison='review/paired_comparison.csv',
        limits='Fresh24 synthetic inputs comprise8 paired factual scenarios and8 self-contained controls, not24 independent human conversations. Validation reuses48 histories in two layouts plus32 self-contained controls. Exact standard-refusal and lexical matches are diagnostics, not semantic accuracy. V1/V2 had authored speaker-reference errors; V3 resets from original weights. No matched control rerun, independent human evaluation, locked-test result or automatic promotion.')
    save('summary.json', summary)
    files = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(ROOT)
        if '__pycache__' in rel.parts or any(part=='selected' or part.startswith('step_') for part in rel.parts[:-1]):
            continue
        if p.name.startswith('results_archive'):
            continue
        if p.suffix in {'.json', '.py', '.csv', '.sbatch', '.out'}:
            files.append(p)
    archive = ROOT/'results_only_v1.tgz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    record = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive), files=len(files), weights_included=False)
    save('results_archive.json', record)
    print(json.dumps(dict(status=summary['status'], selected_step=step, archive=record,
        main_metrics=summary['main_metrics'])), flush=True)


if __name__ == '__main__':
    main()
