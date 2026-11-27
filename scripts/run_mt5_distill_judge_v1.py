"""Run the team's unchanged judge prompt with matched, deduplicated distillation replies."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from statistics import mean
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable_sha(d):return hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def verify_packet(config,public,mapping):
    assert len(public)==config['unique_candidates'] and len(mapping)==len(public)
    assert len(config['personas'])==10 and len({p['id'] for p in config['personas']})==10
    seen=set();coverage=defaultdict(set);exact_keys=set()
    for row in public:
        assert set(row)=={'schema_version','candidate_id','conversation_id','conversation','incoming_message','style','reply'}
        cid=row['candidate_id'];assert cid not in seen and cid in mapping;seen.add(cid)
        m=mapping[cid];assert row['style']==m['style'] and row['conversation_id']==m['conversation_id']
        assert row['style'] in ['STANDARD','LOCAL'] and row['reply'].strip()
        assert row['incoming_message']['speaker']=='peer'
        for t in [*row['conversation'],row['incoming_message']]:
            assert set(t)=={'speaker','text'} and t['speaker'] in ['peer','user'] and t['text'].strip()
        key=(m['item_id'],row['style'],row['reply']);assert key not in exact_keys;exact_keys.add(key)
        assert len(m['models'])==len(set(m['models']))
        for model in m['models']:
            assert model in config['models']
            cell=(m['item_id'],row['style']);assert cell not in coverage[model];coverage[model].add(cell)
    expected=next(iter(coverage.values()))
    assert len(expected)==100 and len({x[0] for x in expected})==50
    assert set(coverage)==set(config['models']) and all(c==expected for c in coverage.values())
    return coverage

def import_judge():
    spec=importlib.util.spec_from_file_location('groupmate_judge',ROOT/'scripts/groupmate_judge.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def tasks_for(public,personas):
    tasks=[(row,persona) for row in public for persona in personas]
    random.Random(425).shuffle(tasks)
    return tasks

def load_journal(path,run_id,allowed,judge):
    latest={}
    for line in Path(path).read_text().splitlines() if Path(path).exists() else []:
        row=json.loads(line);key=(row['candidate_id'],row['persona_id'])
        if row['run_id']!=run_id or key not in allowed or latest.get(key,{}).get('status')=='ok':
            raise ValueError('Journal/configuration mismatch or duplicate completed judgment')
        if row['status']=='ok':judge.validate_rating(row['rating'])
        elif row['status']!='error' or row['rating'] is not None:raise ValueError('Invalid failed-rating record')
        latest[key]=row
    return latest

def expand_and_summarise(config,public,mapping,latest):
    """Preserve all system mappings and average personas before conversations."""
    tasks=tasks_for(public,config['personas']);expected={(r['candidate_id'],p['id']) for r,p in tasks}
    if not all(latest.get(k,{}).get('status')=='ok' for k in expected):
        raise ValueError('Incomplete matched run; do not rank partial averages')
    expanded=[];cells=defaultdict(list)
    for r,p in tasks:
        record=latest[(r['candidate_id'],p['id'])];meta=mapping[r['candidate_id']]
        for model in meta['models']:
            item={'model':model,'item_id':meta['item_id'],'style':r['style'],'candidate_id':r['candidate_id'],
                  'persona_id':p['id'],'rating':record['rating'],'shared_exact_reply_rating':len(meta['models'])>1}
            expanded.append(item);cells[(model,r['style'],meta['item_id'])].append(record['rating'])
    assert len(expanded)==config['expanded_ratings']==5000
    case_scores={}
    for key,ratings in cells.items():
        assert len(ratings)==10
        case_scores[key]={metric:mean(float(r[metric]) for r in ratings) for metric in ['appropriateness','naturalness','style_fit']}
        case_scores[key]['would_send_pct']=100*mean(r['would_send'] for r in ratings)
        case_scores[key]['uncertain_pct']=100*mean(r['uncertain'] for r in ratings)
    summaries=[]
    for model in config['models']:
        for style in ['STANDARD','LOCAL']:
            group=[v for (m,s,_),v in case_scores.items() if m==model and s==style];assert len(group)==50
            summaries.append({'model':model,'style':style,'conversations':50,'persona_ratings':500,
                              **{k:mean(v[k] for v in group) for k in group[0]}})
    comparisons=[]
    for model in config['models'][1:]:
        for style in ['STANDARD','LOCAL']:
            ids=sorted(i for m,s,i in case_scores if m==model and s==style)
            for metric in ['appropriateness','naturalness','style_fit','would_send_pct']:
                diffs=[case_scores[(model,style,i)][metric]-case_scores[('unchanged',style,i)][metric] for i in ids]
                rng=random.Random(425);samples=sorted(mean(diffs[rng.randrange(50)] for _ in range(50)) for _ in range(2000))
                comparisons.append({'model':model,'baseline':'unchanged','style':style,'metric':metric,
                                    'paired_mean_difference':mean(diffs),'ci95_low':samples[49],'ci95_high':samples[1949],
                                    'bootstrap_unit':'conversation; all persona/system ratings kept together','conversations':50})
    return expanded,summaries,comparisons

def verify_model_blobs(config,out):
    store=ROOT/'model_store';manifest=store/'manifests/registry.ollama.ai/library/gemma4/12b'
    assert sha(manifest)==config['expected_model_digest'],'Judge manifest is not the historical pinned version'
    d=read(manifest);entries=[d['config'],*d['layers']];checks=[]
    for v in entries:
        p=store/'blobs'/v['digest'].replace(':','-');assert p.stat().st_size==v['size']
        assert sha(p)==v['digest'].split(':')[1],str(p)
        checks.append({'digest':v['digest'],'size':p.stat().st_size})
    write(out/'model_blobs_verified.json',{'manifest_sha256':sha(manifest),'blobs':checks})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','main','summary','check'],required=True)
    args=parser.parse_args();config=read(ROOT/'config.json');public=read(ROOT/'judge_inputs.json');mapping=read(ROOT/'PRIVATE_mapping.json')
    verify_packet(config,public,mapping)
    for name,h in read(ROOT/'payload_manifest.json').items():assert sha(ROOT/name)==h,('Frozen payload changed',name)
    if args.mode=='check':print('Frozen matched packet verified; no judge/model calls.');return
    judge=import_judge();reference=read(ROOT/'reference_judge_run.json')
    assert judge.OPTIONS==reference['options']==config['options'] and judge.OLLAMA_THINK==config['think']
    assert judge.SCHEMA==reference['response_schema']
    rubric=(ROOT/'data/evaluation/judge_rubric.md').read_text();assert rubric==reference['rubric']
    all_tasks=tasks_for(public,config['personas']);allowed={(r['candidate_id'],p['id']) for r,p in all_tasks}
    identity={'config_sha256':sha(ROOT/'config.json'),'payload_manifest_sha256':sha(ROOT/'payload_manifest.json'),
              'judge_model':config['judge_model'],'model_digest':config['expected_model_digest'],
              'options':config['options'],'think':config['think'],'workers':config['workers'],
              'groupmate_judge_script_sha256':config['groupmate_judge_script_sha256']}
    run_id=stable_sha(identity);out=ROOT/'results';out.mkdir(exist_ok=True)
    from filelock import FileLock
    with FileLock(str(out/'.lock'),timeout=0):
        manifest=out/'run.json'
        if manifest.exists():assert read(manifest)==identity,'Frozen evaluation run identity changed'
        else:write(manifest,identity)
        latest=load_journal(out/'ratings.jsonl',run_id,allowed,judge)
        if args.mode=='summary':
            expanded,summary,comparisons=expand_and_summarise(config,public,mapping,latest)
        else:
            if args.mode=='main':
                approval=read(ROOT/'pilot_review_approved.json')
                assert approval['approved'] and approval['config_sha256']==sha(ROOT/'config.json')
                assert read(out/'pilot_complete.json')['all_valid_complete']
            verify_model_blobs(config,out)
            base=judge.local_url(os.environ['OLLAMA_BASE_URL'])
            prepared=judge.prepare_judge({'base':base,'model':config['judge_model']})
            assert prepared['model_digest']==config['expected_model_digest'],prepared
            assert prepared['options']==config['options'] and prepared['think']==config['think']
            todo=all_tasks if args.mode=='main' else [(r,p) for r,p in all_tasks if r['candidate_id'] in config['pilot_candidate_ids'] and p['id'] in config['pilot_persona_ids']]
            phase_keys={(r['candidate_id'],p['id']) for r,p in todo}
            todo=[(r,p) for r,p in todo if latest.get((r['candidate_id'],p['id']),{}).get('status')!='ok']
            started=time.monotonic();job=os.environ.get('SLURM_JOB_ID','local');calls=0
            print(json.dumps({'mode':args.mode,'remaining_requests':len(todo),'all_unique_requests':len(allowed),'judge':prepared,'workers':config['workers']}),flush=True)
            def one(task):
                r,p=task;begin=time.monotonic();record={'run_id':run_id,'candidate_id':r['candidate_id'],'persona_id':p['id'],'status':'error','rating':None,'attempts':0,'job_id':job}
                for attempt in range(config['retry_attempts']):
                    record['attempts']+=1
                    try:
                        rating,metadata=judge.judge_one(base,config['judge_model'],p,r,rubric)
                        if metadata.get('prompt_eval_count') is not None and metadata['prompt_eval_count']>=config['options']['num_ctx']:
                            raise ValueError('Judge prompt exhausted context window')
                        record.update(status='ok',rating=rating,ollama=metadata,error=None);break
                    except (OSError,ValueError,TypeError) as exc:record['error']=str(exc)[:500]
                record['seconds']=round(time.monotonic()-begin,4);return record
            with (out/'ratings.jsonl').open('a') as handle,ThreadPoolExecutor(max_workers=config['workers']) as pool:
                futures=[pool.submit(one,t) for t in todo]
                for future in as_completed(futures):
                    record=future.result();key=(record['candidate_id'],record['persona_id'])
                    handle.write(json.dumps(record,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
                    latest[key]=record;calls+=1
                    if calls==1 or calls%10==0 or calls==len(todo):
                        progress={'mode':args.mode,'run_id':run_id,'job_id':job,'completed_unique':sum(v['status']=='ok' for v in latest.values()),
                                  'expected_unique':len(allowed),'errors':sum(v['status']=='error' for v in latest.values()),'elapsed_seconds':time.monotonic()-started,
                                  'phase_requests_completed':calls,'phase_requests_submitted':len(todo)}
                        write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            ps=judge.request_json(base,'/api/ps');loaded=[m for m in ps.get('models',[]) if m.get('name') in [config['judge_model'],config['judge_model']+':latest']]
            assert loaded and all(m.get('size_vram',0)>0 for m in loaded),'Judge was not running on an allocated GPU'
            runtime={'job_id':job,'mode':args.mode,'judge':prepared,'loaded_models':loaded,'ollama_num_parallel':os.environ.get('OLLAMA_NUM_PARALLEL'),
                     'nvidia_smi':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv'],capture_output=True,text=True).stdout}
            write(out/f'runtime_{job}.json',runtime)
            phase_ok=all(latest.get(k,{}).get('status')=='ok' for k in phase_keys)
            if args.mode=='pilot':
                durations=[latest[k]['seconds'] for k in phase_keys]
                result={'config_sha256':sha(ROOT/'config.json'),'run_id':run_id,'all_valid_complete':phase_ok,'requests':len(phase_keys),
                        'wall_seconds':time.monotonic()-started,'mean_request_seconds':mean(durations),'max_prompt_tokens':max(latest[k].get('ollama',{}).get('prompt_eval_count',0) or 0 for k in phase_keys),
                        'gpu_confirmed':True,'model_digest':prepared['model_digest'],'no_score_threshold_used':True}
                write(out/'pilot_complete.json',result);print(json.dumps(result),flush=True)
                if not phase_ok:raise ValueError('Pilot has failed/invalid responses')
                return
            expanded,summary,comparisons=expand_and_summarise(config,public,mapping,latest)
        with (out/'expanded_ratings.jsonl').open('w') as f:
            for row in expanded:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        for name,rows in [('summary.csv',summary),('paired_differences.csv',comparisons)]:
            with (out/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write(out/'summary.json',summary);write(out/'paired_differences.json',comparisons)
        write(out/'complete.json',{'status':'complete','run_id':run_id,'unique_judgments':len(allowed),'expanded_ratings':len(expanded),'conversations':50,'models':5,'styles':2,'personas':10,'all_valid_complete':True,'independent_human_evaluation':False,'test_set_used':False})
        print(json.dumps({'complete':True,'summary':summary}),flush=True)

if __name__=='__main__':main()
