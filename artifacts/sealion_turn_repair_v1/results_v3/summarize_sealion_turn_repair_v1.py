"""Collect complete turn-repair studies, comparisons and adapter checksums."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
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


def compare(paths,name):
    reports={key:read(path) for key,path in paths.items()}
    assert all(r['status']=='complete' for r in reports.values())
    ids=[row['item_id'] for row in next(iter(reports.values()))['outputs']]
    assert all([row['item_id'] for row in r['outputs']]==ids for r in reports.values())
    rows=[]
    for index,ident in enumerate(ids):
        source=next(iter(reports.values()))['outputs'][index]
        assert all(r['outputs'][index]['context']==source['context'] and r['outputs'][index]['message']==source['message'] for r in reports.values())
        rows.append(dict(item_id=ident,scope=source['scope'],context=list(zip(source['speakers'],source['context'],strict=True)),
            message=source['message'],reference=source.get('reference'),
            methods={key:{k:report['outputs'][index][k] for k in ['english','singlish','english_status','singlish_status']}
                for key,report in reports.items()}))
    save(f'review/{name}.json',rows)
    fields=['item_id','scope','context','message','reference']+[key+'_'+style for key in reports for style in ['english','singlish']]
    with (ROOT/f'review/{name}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow(dict(item_id=row['item_id'],scope=row['scope'],context=json.dumps(row['context'],ensure_ascii=False),
                message=row['message'],reference=row['reference'],
                **{key+'_'+style:out[style] for key,out in row['methods'].items() for style in ['english','singlish']}))
    return rows


def main():
    assert Path.home().name=='hm.teoh.2024'
    v1=read('results.json');v2=read('layout_v2/results.json')
    assert v1['status']==v2['status']=='complete'
    assert v1['source_adapters_unchanged'] and v2['source_adapters_unchanged']
    assert v1['matched_initial_weights_and_schedules']
    assert read('loader_audit.json')['differences']==[]
    paths={'original':'outputs/baseline/evaluation.json'}
    snapshot=read('snapshot_review.json')
    for arm in ['supervised','contrastive']:
        assert snapshot['reviews'][arm]['step']==40
        paths[arm+'40']=f'outputs/{arm}/best_target_validation_exploratory/evaluation.json'
        paths[arm+'120']=f'outputs/{arm}/final120/evaluation.json'
        selected=read(f'outputs/{arm}/selected/evaluation.json')
        if v1['arms'][arm]['selected_step']==0:
            assert [(r['english'],r['singlish']) for r in selected['outputs']]==[(r['english'],r['singlish']) for r in read(paths['original'])['outputs']]
    rows=compare(paths,'turn_repair_comparison')
    layout_paths={'original':'layout_v2/outputs/original_canonical/evaluation.json',
        'untrained_layout':'layout_v2/outputs/untrained_layout/evaluation.json',
        'trained_selected':'layout_v2/outputs/supervised/selected/evaluation.json'}
    layout_paths['trained_best_target_validation_exploratory']='layout_v2/outputs/supervised/best_target_validation_exploratory/evaluation.json'
    if v2['arms']['supervised']['selected_step']!=120:
        layout_paths['trained120']='layout_v2/outputs/supervised/final120/evaluation.json'
    layout=compare(layout_paths,'layout_comparison')
    policy=read('context_policy_v4/results.json')
    assert policy['status']=='complete' and policy['source_adapters_unchanged']
    assert policy['tests_passed']==31 and policy['cases']==71
    fields=['item_id','scope','context','message','rule','original_english','english','original_singlish','singlish','singlish_status','english_provenance','singlish_provenance']
    with (ROOT/'review/context_policy_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in policy['outputs']:
            writer.writerow({**{k:row.get(k) for k in fields if k not in {'context','rule'}},
                'context':json.dumps(list(zip(row['speakers'],row['context'],strict=True)),ensure_ascii=False),
                'rule':row['context_policy']['rule']})
    save('review/context_policy_comparison.json',policy['outputs'])
    rng=random.Random(8411)
    packet=[];key={}
    for prefix,group in [('turn',rows),('layout',layout)]:
        for row in group:
            unique={}
            for name,out in row['methods'].items():unique.setdefault((out['english'],out['singlish']),[]).append(name)
            values=list(unique);rng.shuffle(values)
            labels={chr(65+i):dict(english=value[0],singlish=value[1]) for i,value in enumerate(values)}
            ident=prefix+'-'+row['item_id']
            packet.append(dict(review_id=ident,context=row['context'],incoming=row['message'],options=labels,
                criteria=['Answers the latest incoming message','Preserves explicit actors/facts/constraints','Singlish preserves the English draft','Natural sendable wording'],human_ratings=None))
            key[ident]={chr(65+i):unique[value] for i,value in enumerate(values)}
    save('review/shuffled_review_packet.json',dict(seed=8411,scope='Labels shuffled for review; author has already seen outputs. Not independent blind human evaluation.',cases=packet))
    save('review/shuffled_review_key.json',key)
    inventory=[]
    for prefix,arms in [('', ['supervised','contrastive']),('layout_v2/', ['supervised'])]:
        for arm in arms:
            for label in ['selected','step_0040','step_0080','step_0120']:
                folder=ROOT/prefix/'outputs'/arm/label
                inventory.append(dict(study=prefix or 'canonical_v1',arm=arm,label=label,path=str(folder),
                    bytes=(folder/'adapter_model.safetensors').stat().st_size,
                    sha256=sha(folder/'adapter_model.safetensors'),config_sha256=sha(folder/'adapter_config.json')))
    save('checkpoint_inventory.json',inventory)
    quota_limit=int(os.getxattr(Path.home(),'ceph.quota.max_bytes'))
    quota_used=int(os.getxattr(Path.home(),'ceph.dir.rbytes'))
    summary=dict(status='all_runs_complete',account='hm.teoh.2024',root=str(ROOT),
        jobs=dict(turn_training=358042,loader_audit=358044,snapshot_review=358047,layout_training=358056,layout_snapshot_review=358059,
            generated_validation=358060,context_policy_prototype=358061,context_policy_v2_failed_launch=358068,
            context_policy_v2=358074,context_policy_v3=358077),
        source_adapters_unchanged=True,default_changed=False,human_review_pending=True,
        original_team_sha256=read('audit.json')['original_team_sha256'],locked_test_read=False,
        preflight=read('preflight.json'),layout_preflight=read('layout_v2/preflight.json'),
        canonical_training=v1,layout_training=v2,snapshot_review=snapshot,layout_snapshot_review=read('layout_v2/snapshot_review.json'),
        loader_audit=read('loader_audit.json'),
        generated_validation_review=read('review/generated_validation_scores.json'),
        generated_validation_decision=read('review/generated_validation_decision.json'),
        context_policy={k:v for k,v in policy.items() if k!='outputs'},
        user_case_context_policy=next(row for row in policy['outputs'] if row['item_id']=='USER-CALL-001'),
        context_policy_prototype_note='First prototype unnecessarily rewrote an already-correct route-map answer and introduced a wh-question. Fixed preservation and rewrite-act check. Ten new smoke cases exposed generic chat-word overlap supporting an invented topic; excluded communication words from topic support. A subsequent CPU-only check rejects a rewrite introducing an unprovided past conversation; one additional English fallback, no new inference/training. All revisions/results/payloads retained. Smoke cases are development, not fresh independent test.',
        inference_variation_note='Official/study loaders exactly match on A40; exact original user reply differs across A40/L40S. All training arm comparisons use one L40S. This is not a source checkpoint update.',
        checkpoints=len(inventory),new_trained_snapshots=9,
        comparisons=['review/turn_repair_comparison.csv','review/layout_comparison.csv','review/context_policy_comparison.csv'],
        user_case_canonical=next(row for row in rows if row['item_id']=='USER-CALL-001'),
        user_case_layout=next(row for row in layout if row['item_id']=='USER-CALL-001'),
        quota_bytes=dict(limit=quota_limit,used=quota_used,free=quota_limit-quota_used),
        conclusion='No reliable neural upgrade: all three training arms retainedstep0 by loss guard; generated validation review also found no clear gain over original. Optional context policy supplies labelled deterministic English anchors plus original neural Singlish rewriting or labelled English fallback. Limited rule coverage; not proof the standalone model generalizes.',
        note='Validation loss and contrast preference are not generated accuracy. New scenarios are assistant authored; previous challenge became development after inspection. No independent human or locked-test claim.')
    save('summary.json',summary)
    archive=ROOT/'results_only_v3.tgz'
    included=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(ROOT)
        if '__pycache__' in rel.parts or any(part=='selected' or part.startswith('step_') for part in rel.parts[:-1]):continue
        if p.name.startswith('results_archive'):continue
        if p.suffix in {'.json','.py','.csv','.sbatch','.out'}:included.append(p)
    with tarfile.open(archive,'w:gz') as tar:
        for p in included:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    record=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(included),weights_included=False)
    save('results_archive_v3.json',record)
    print(json.dumps(dict(status=summary['status'],archive=record,user_case=summary['user_case_context_policy'],quota=summary['quota_bytes'])),flush=True)


if __name__=='__main__':main()
