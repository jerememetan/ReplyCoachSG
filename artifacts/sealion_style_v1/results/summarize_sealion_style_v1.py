"""Collect completed personal-account experiment results without model weights."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import sealion_style_v1 as style
import sealion_style_guard_v2 as revised


def read(name):return json.loads((ROOT/name).read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    assert Path.home().name=='hm.teoh.2024'
    algorithms=read('outputs/algorithms/results.json')
    training=read('outputs/training/results.json')
    assert all(report['status']=='complete' and report['source_bundle_weights_unchanged'] for report in [algorithms,training])
    assert not algorithms.get('repair25_replay_differences')
    baseline={row['item_id']:row for row in algorithms['outputs']}
    guard_v2=read('review/guard_v2/results.json')
    revised_rows={row['item_id']:row for row in guard_v2['outputs']}
    lexicon=read('outputs/algorithms/style_lexicon.json')
    snapshots={}
    for arm in ['control','copy_loss']:
        assert training['arms'][arm]['optimizer_steps']==100
        assert read(f'outputs/snapshot_review/{arm}/summary.json')['status']=='complete'
        for step in [25,50,100]:
            report=read(f'outputs/snapshot_review/{arm}/step_{step:04d}.json')
            assert report['status']=='complete' and len(report['outputs'])==32
            snapshots[f'{arm}_{step}']={row['item_id']:row for row in report['outputs']}
    ids=list(snapshots['control_25'])
    assert all(set(rows)==set(ids) for rows in snapshots.values())
    comparisons=[]
    for ident in ids:
        original=baseline[ident]
        texts={method:entry['text'] for method,entry in original['methods'].items()}
        texts['combined_guard_v2_copy1']=revised_rows[ident]['methods']['copy1_style0.2']['text']
        for name,rows in snapshots.items():
            assert rows[ident]['standard_anchor']==original['standard_anchor']
            texts[name]=rows[ident]['methods']['beam4']['text']
        comparisons.append(dict(item_id=ident,scope=original['scope'],english=original['standard_anchor'],
            context=original['context'],speakers=original['speakers'],message=original['message'],outputs=texts))
    review=ROOT/'review'
    fields=['item_id','scope','english','context','message']+list(comparisons[0]['outputs'])
    with (review/'all_methods_and_checkpoints.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in comparisons:writer.writerow(dict(item_id=row['item_id'],scope=row['scope'],english=row['english'],
            context=json.dumps(list(zip(row['speakers'],row['context'],strict=True)),ensure_ascii=False),message=row['message'],**row['outputs']))
    save(review/'all_methods_and_checkpoints.json',comparisons)
    short_metrics={}
    for name in comparisons[0]['outputs']:
        scopes={}
        for scope in ['reused_development','assistant_authored_diagnostic']:
            rows=[row for row in comparisons if row['scope']==scope]
            scopes[scope]=dict(cases=len(rows),exact_copies=sum(style.canonical(row['english'])==style.canonical(row['outputs'][name]) for row in rows),
                revised_surface_check_passes=sum(revised.guard(row['english'],row['outputs'][name])['passed'] for row in rows),
                note='Same small review subset. Surface checks are incomplete; these are not human quality scores.')
        short_metrics[name]=scopes
    save(review/'snapshot_metrics.json',short_metrics)
    rng=random.Random(20261108)
    familiar={'SEADUAL-011','SEADUAL-015','SEADUAL-008','SEADUAL-010','SEADUAL-013'}
    selected=[row for row in comparisons if row['item_id'] in familiar]
    selected+=rng.sample([row for row in comparisons if row['scope']=='reused_development' and row['item_id'] not in familiar],5)
    packet=[];key={}
    methods=['beam4','combined_guard_v2_copy1','control_100','copy_loss_25','copy_loss_50','copy_loss_100']
    for i,row in enumerate(selected,1):
        unique={}
        for method in methods:unique.setdefault(row['outputs'][method],[]).append(method)
        options=list(unique);rng.shuffle(options)
        identity=f'TRAINED-STYLE-{i:03d}'
        labels={chr(65+n):text for n,text in enumerate(options)}
        packet.append(dict(review_id=identity,context=list(zip(row['speakers'],row['context'],strict=True)),message=row['message'],
            english=row['english'],options=labels,criteria=['Context-compatible useful reply','English-anchor meaning preservation','Natural restrained Singlish'],
            note='Alternative useful intentions can be judged separately from faithful rewriting. No mandatory particles.'))
        key[identity]=dict(item_id=row['item_id'],mapping={letter:unique[text] for letter,text in labels.items()})
    if not (review/'training_blind_packet.json').exists():
        save(review/'training_blind_packet.json',dict(seed=20261108,cases=packet,human_ratings=None))
        save(review/'training_blind_key.json',key)
    inventory=[]
    for arm in ['control','copy_loss']:
        for name in ['selected','step_0025','step_0050','step_0100']:
            path=ROOT/'outputs/training'/arm/name
            inventory.append(dict(arm=arm,checkpoint=str(path),label=name,selected_step=training['arms'][arm]['selected_step'],
                weights_bytes=(path/'adapter_model.safetensors').stat().st_size,
                weights_sha256=sha(path/'adapter_model.safetensors'),config_sha256=sha(path/'adapter_config.json'),
                base_reference=str(Path.home()/'ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1/base')))
    save(ROOT/'checkpoint_inventory.json',inventory)
    selected_replay={}
    for arm in ['control','copy_loss']:
        selected=read(f'outputs/training/{arm}/evaluation.json')
        assert selected['status']=='complete' and len(selected['outputs'])==136
        differences=[row['item_id'] for row in selected['outputs'] if row['methods']['beam4']['text']!=baseline[row['item_id']]['methods']['beam4']['text']]
        selected_replay[arm]=dict(selected_step=training['arms'][arm]['selected_step'],differences=differences)
    remaining=[]
    for row in guard_v2['outputs']:
        if row['scope']!='reused_development':continue
        chosen=row['methods']['copy1_style0.2']['text']
        if style.copy_features(row['english'],chosen)['unnecessary_copy']:
            valid,_=revised.rank(row['english'],baseline[row['item_id']]['candidate_pool'],lexicon,1.,.2)
            noncopies=[item for item in valid if not style.copy_features(row['english'],item['text'])['exact_copy']]
            remaining.append(dict(item_id=row['item_id'],english=row['english'],guard_passed_noncopy_candidates=len(noncopies)))
    quota_limit=int(os.getxattr(Path.home(),'ceph.quota.max_bytes'));quota_used=int(os.getxattr(Path.home(),'ceph.dir.rbytes'))
    summary=dict(status='all_experiments_complete',account='hm.teoh.2024',root=str(ROOT),
        jobs=dict(preflight=357996,algorithms=357998,training=357999,control_snapshot_review=358014,copy_snapshot_review=358016),
        data=read('data/data_manifest.json'),tests_passed=11,guard_followup_regressions=7,
        source_bundle_weights_unchanged=True,default_changed=False,human_review_pending=True,
        original_repair25_exact_copies=10,main_development_cases=126,
        v1_metrics=algorithms['metrics'],v2_sweeps=guard_v2['summary']['sweeps'],
        training={arm:{key:training['arms'][arm][key] for key in ['optimizer_steps','selected_step','initial_validation','history','copy_margin_active_steps','frozen_base_unchanged','ordinary_exposures','auxiliary_pair_exposures']} for arm in ['control','copy_loss']},
        selected_step0_replays=selected_replay,snapshot_review_cases=32,checkpoint_count=len(inventory),
        remaining_copies=remaining,
        conclusion='No automatic upgrade: the initial reranker increases copies; corrected guard/higher penalties match baseline10/126. Both training arms retainstep0 by validation guard. Snapshots retained for human assessment; copy rate alone is not quality.',
        quota_bytes=dict(limit=quota_limit,used=quota_used,free=quota_limit-quota_used),
        review_csv='review/all_methods_and_checkpoints.csv',training_review_packet='review/training_blind_packet.json',
        analysis_script_sha256=sha(Path(__file__)))
    save(ROOT/'summary.json',summary)
    # Preserve full reports and frozen code, omit adapters and source base weights.
    archive=ROOT/'results_only_v1.tgz'
    include=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.is_symlink() or path==archive:continue
        rel=path.relative_to(ROOT)
        if '__pycache__' in rel.parts or any(part=='selected' or part.startswith('step_') for part in rel.parts[:-1]):continue
        if path.suffix in {'.json','.csv','.py','.out','.sbatch'}:include.append(path)
    with tarfile.open(archive,'w:gz') as tar:
        for path in include:tar.add(path,arcname=str(path.relative_to(ROOT)))
    record=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(include),weights_included=False)
    save(ROOT/'results_archive.json',record)
    print(json.dumps(dict(status=summary['status'],selected_replays=selected_replay,checkpoints=len(inventory),
        remaining_copy_cases=len(remaining),remaining_without_guard_passed_noncopy=sum(row['guard_passed_noncopy_candidates']==0 for row in remaining),
        archive=record,quota=summary['quota_bytes'])),flush=True)


if __name__=='__main__':main()
