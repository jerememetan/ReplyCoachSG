"""New, separately frozen six-system regrade of the previously inspected 60 test cases."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import random
from statistics import mean
import subprocess
import sys
import time

from filelock import FileLock
from replycoach.current_final import make_current_packet,VERSION_LABELS
from replycoach.final_test import validate_test_rows
from replycoach.io import ROOT,read_jsonl
from replycoach.judging import write_judging_packet
from replycoach.transformer_comparison import task_key

EXPECTED_GEMMA='6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def digest(path):
    result=sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def report_results(out,public,mapping,latest,personas,judge,identity):
    summary=judge.summarise(list(latest.values()),mapping,len(personas))
    if len(summary)!=12 or any(r['rated']!=600 or r['failed'] for r in summary):
        raise ValueError('Need twelve complete 60-conversation/style groups')
    cells=defaultdict(list)
    for (cid,_),row in latest.items():
        meta=mapping[cid];cells[meta['system'],meta['item_id'],meta['style']].append(row['rating'])
    for row in summary:
        values=[v for (model,_,style),ratings in cells.items()
                if model==row['system'] and style==row['style'] for v in ratings]
        row['problem_counts']=dict(Counter(v['problem'] for v in values))
        row['uncertain_count']=sum(v['uncertain'] for v in values)
    items=sorted({r['item_id'] for r in mapping.values()})
    rng=random.Random(425);samples=[rng.choices(range(60),k=60) for _ in range(5000)]
    paired=[]
    for after,before in [('V6','V5'),('V7','V6')]:
        for style in ['STANDARD','LOCAL']:
            for metric in [*judge.SCORES,'would_send']:
                diffs=[mean(r[metric] for r in cells[after,item,style])-
                       mean(r[metric] for r in cells[before,item,style]) for item in items]
                boot=sorted(sum(diffs[i] for i in indices)/60 for indices in samples)
                paired.append(dict(after=after,before=before,style=style,metric=metric,
                                   difference=mean(diffs),ci95=[boot[125],boot[4875]]))
    result=dict(completed=True,record_type='current_pipeline_regrade',evaluation_split='test',
        conversations=60,nominal_rating_cells=7200,version_labels=VERSION_LABELS,
        rubric='current fixed rubric',judge=identity,summary=summary,paired=paired,
        provenance_counts=dict(Counter(r['provenance']['kind'] for r in latest.values())),
        previous_test_ratings_reused=False,
        limitation='Previously inspected test split, explicitly re-evaluated after validation-based SEA-LION selection. Correlated AI personas, not independent human ratings. No tuning from these results.')
    save(out/'result.json',result)
    lines=['# Current six-system pipeline re-evaluation','',
        '60 previously inspected test conversations; one current rubric, same ten Gemma personas. '
        'V5 uses SEA-LION; no old-rubric grades are reused.','',
        '| System | Style | Appropriateness | Naturalness | Style fit | Would send |',
        '|---|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['system']} | {row['style']} | {row['appropriateness']:.3f} | "
            f"{row['naturalness']:.3f} | {row['style_fit']:.3f} | {row['would_send_pct']:.1f}% |")
    lines+=['','Version labels:']+[f'- {k}: {v}.' for k,v in VERSION_LABELS.items()]
    lines+=['','V3 is not included: its supplied validation exports do not cover these 60 test cases.',
        '', 'Within-run exact-task/persona reuse: '+json.dumps(result['provenance_counts']),
        '',result['limitation'],'','## Paired pipeline changes','',json.dumps(paired,indent=2),'']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def run_current(out,resume=False):
    if out.exists()!=resume:
        raise ValueError('Use a new output folder, or --resume for an existing current-profile run')
    out.mkdir(parents=True,exist_ok=resume)
    with FileLock(str(out/'.lock'),timeout=0):
        try:_run(out,resume)
        except Exception as exc:
            save(out/'progress.json',dict(phase='failed',error=f'{type(exc).__name__}: {exc}'))
            raise


def _run(out,resume):
    # A cluster handoff manifest verifies data, code and private model bytes before use.
    handoff=ROOT/'current_final_input_manifest.json'
    if handoff.exists():
        for name,checksum in read(handoff)['files'].items():
            if digest(ROOT/name)!=checksum:raise ValueError('Transferred file changed: '+name)
    bundle=ROOT/'artifacts/model_handoff_v1/replycoach_sealion_repair25_v1'
    for name,entry in read(bundle/'manifest.json')['files'].items():
        if digest(bundle/name)!=entry['sha256']:
            raise ValueError('SEA-LION bundle changed: '+name)
    spec=importlib.util.spec_from_file_location('current_judge',ROOT/'scripts/judge_replies.py')
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    config=judge.judge_settings(judge.read_env(ROOT/'.env'));identity=judge.prepare_judge(config)
    if identity['model_digest']!=EXPECTED_GEMMA:raise ValueError('Use the exact agreed Gemma model')
    personas=judge.load_personas(ROOT/'data/evaluation/personas.csv')
    if len(personas)!=10:raise ValueError('Expected ten fixed personas')
    rubric=judge.RUBRIC_PATH.read_text(encoding='utf-8')
    files=[ROOT/p for p in ['data/dev/train.jsonl','data/dev/validation.jsonl','data/test/test.jsonl',
        'data/split/manifest.json','data/evaluation/personas.csv','data/evaluation/judge_rubric.md',
        'data/lexicons/local_terms.json','scripts/run_final_test.py','scripts/current_final_runner.py',
        'scripts/export_current_final.py','scripts/judge_replies.py','artifacts/rnn_v4_task.pt',
        'artifacts/model_handoff_v1/replycoach_sealion_repair25_v1/manifest.json']]
    files+=sorted((ROOT/'src/replycoach').rglob('*.py'))
    for folder in ['artifacts/minilm_retrieval','artifacts/reranker_qwen/pretrained',
                   'artifacts/reranker_qwen/dd_result_20261029/stage2_reviewed/best']:
        models=[p for p in (ROOT/folder).rglob('*') if p.is_file()]
        if not models:raise FileNotFoundError('Missing pinned model folder '+folder)
        files+=models
    freeze=dict(record_type='current_pipeline_regrade',evaluation_split='test',conversations=60,
        version_labels=VERSION_LABELS,v5='SEA-LION 3B',v7_seed=425,nominal_rating_cells=7200,
        current_rubric=rubric,personas=personas,judge=identity,
        files={p.relative_to(ROOT).as_posix():digest(p) for p in files},
        previous_test_exposure=True,previous_test_ratings_reused=False,no_test_tuning=True,
        history_policy='Separate run. Do not overwrite old exports, grades or freeze files.')
    if resume:
        if read(out/'freeze.json')!=freeze:raise ValueError('Frozen current-profile inputs changed')
    else:save(out/'freeze.json',freeze)
    assignment=read(ROOT/'data/split/manifest.json')['assignment']
    expected={i for i,s in assignment.items() if s=='test'}
    if len(expected)!=60:raise ValueError('Expected the official 60 test cases')
    items=validate_test_rows(read_jsonl(ROOT/'data/test/test.jsonl'),expected,
        read_jsonl(ROOT/'data/dev/train.jsonl')+read_jsonl(ROOT/'data/dev/validation.jsonl'))
    journal=out/'outputs.jsonl'
    exports=read_jsonl(journal) if journal.exists() else []
    if len(exports)<60:
        subprocess.check_call([sys.executable,'-u',str(ROOT/'scripts/export_current_final.py'),str(out)],cwd=ROOT)
    exports=read_jsonl(journal)
    if len(exports)!=60 or len({r['item_id'] for r in exports})!=60:
        raise ValueError('Generation journal does not cover all sixty cases')
    outputs={r['item_id']:r['output'] for r in exports}
    public,private=make_current_packet(items,outputs)
    packet=out/'packet'
    if not packet.exists():write_judging_packet(packet,public,private)
    loaded,mapping,hashes=judge.load_packet(packet)
    if loaded!=public:raise ValueError('Saved judging packet differs from current exports')
    receipt=dict(**identity,personas=personas,rubric=rubric,response_schema=judge.SCHEMA,
        provider='ollama',calibration=False,packet_hashes=hashes,freeze_sha256=digest(out/'freeze.json'))
    if (out/'judge_run.json').exists() and read(out/'judge_run.json')!=receipt:
        raise ValueError('Current grader identity changed')
    save(out/'judge_run.json',receipt)
    run_id=sha256(json.dumps(receipt,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    by_id={r['candidate_id']:r for r in public};personas_ids={p['id'] for p in personas}
    ratings=out/'ratings.jsonl';latest={};cache={}
    for row in read_jsonl(ratings) if ratings.exists() else []:
        key=row['candidate_id'],row['persona_id']
        if row['run_id']!=run_id or key[0] not in by_id or key[1] not in personas_ids:
            raise ValueError('Saved rating journal mismatch')
        if latest.get(key,{}).get('status')=='ok':raise ValueError('Duplicate completed rating')
        if row['status']=='ok':
            judge.validate_rating(row['rating'])
            cache[task_key(by_id[key[0]],key[1])]=(row['rating'],key[0])
        latest[key]=row
    started=time.monotonic();calls=0;reused=0
    runtime=dict(job_id=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        ollama_version=judge.request_json(config['base'],'/api/version'),
        started_at=datetime.now(timezone.utc).isoformat())
    save(out/'cluster_runtime.json',runtime)
    with ratings.open('a',encoding='utf-8') as handle:
        for candidate in public:
            for persona in personas:
                key=candidate['candidate_id'],persona['id']
                if latest.get(key,{}).get('status')=='ok':continue
                tick=time.monotonic()
                row=dict(run_id=run_id,candidate_id=key[0],persona_id=key[1],status='error',rating=None,
                    execution='cluster',job_id=runtime['job_id'],node=runtime['node'],
                    attempts=latest.get(key,{}).get('attempts',0),created_at=datetime.now(timezone.utc).isoformat())
                cached=cache.get(task_key(candidate,persona['id']))
                if cached:
                    row.update(status='ok',rating=cached[0],provenance={'kind':'current_run','candidate_id':cached[1]});reused+=1
                else:
                    for attempt in range(2):
                        calls+=1;row['attempts']+=1
                        try:
                            value,timing=judge.judge_one(config['base'],config['model'],persona,candidate,rubric)
                            row.update(status='ok',rating=value,ollama=timing,provenance={'kind':'new_gemma_call'})
                            cache[task_key(candidate,persona['id'])]=(value,key[0]);break
                        except (OSError,ValueError,TypeError) as exc:row['error']=str(exc)[:500]
                row['seconds']=round(time.monotonic()-tick,3)
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
                latest[key]=row
                progress=dict(phase='judging',completed=sum(r['status']=='ok' for r in latest.values()),total=7200,
                    errors=sum(r['status']!='ok' for r in latest.values()),api_calls_this_process=calls,
                    reused_this_process=reused,last_call_seconds=row['seconds'],job_id=runtime['job_id'],
                    node=runtime['node'],elapsed_seconds=round(time.monotonic()-started,2))
                save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    if len(latest)!=7200 or any(r['status']!='ok' for r in latest.values()):
        raise RuntimeError('Incomplete matched regrade; resume before reporting results')
    report_results(out,public,mapping,latest,personas,judge,identity)
    save(out/'progress.json',dict(phase='completed',completed=7200,total=7200,errors=0,job_id=runtime['job_id']))
