"""Freeze the team's judge protocol and matched warm-distillation validation replies."""
from pathlib import Path
import csv
import hashlib
import json
import random
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/mt5_distill_judge_v1'
STAGE=OUT/'stage'
REFERENCE=OUT/'groupmate_reference'
ARMS=['unchanged','sealion_ce','sealion_kd','qwen_ce','qwen_kd']
REMOTE='/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG_mt5_distill_judge_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def build_packet(rows, queries):
    """Deduplicate only exact text within one conversation and requested style."""
    rng=random.Random(425)
    by_id={r['item_id']:r for r in rows}
    queries=sorted(queries,key=lambda r:r['id']);rng.shuffle(queries)
    groups=[]
    for i,q in enumerate(queries,1):
        r=by_id[q['id']]
        assert r['message']==q['message']['text']
        assert r['context']==[x['text'] for x in q['session_memory']]
        assert r['speakers']==[x['speaker'] for x in q['session_memory']]
        for style,stage in [('STANDARD','english'),('LOCAL','singlish')]:
            by_text={}
            for model in ARMS:
                reply=r[model+'_'+stage]
                assert isinstance(reply,str) and reply.strip()
                by_text.setdefault(reply,[]).append(model)
            for reply,models in by_text.items():
                groups.append({'public':{'schema_version':1,'conversation_id':f'Q{i:06d}',
                    'conversation':[{'speaker':x['speaker'],'text':x['text']} for x in q['session_memory']],
                    'incoming_message':{'speaker':'peer','text':q['message']['text']},'style':style,'reply':reply},
                    'private':{'item_id':q['id'],'models':models}})
    rng.shuffle(groups)
    for i,g in enumerate(groups,1):g['public']['candidate_id']=f'C{i:06d}'
    return groups

def main():
    if STAGE.exists():raise FileExistsError('Preserve the already frozen evaluation stage')
    comparison=ROOT/'artifacts/mt5_warm_distill_v1/full_results_20261121/selected_comparison.json'
    rows=json.loads(comparison.read_text())
    queries=[json.loads(l) for l in (ROOT/'data/dev/validation.jsonl').read_text().splitlines() if l.strip()]
    assert len(queries)==50 and all(q['split']=='dev' and q['privacy_ok'] for q in queries)
    packet=build_packet(rows,queries)
    assert len(packet)==339 and sum(len(g['private']['models']) for g in packet)==500
    STAGE.mkdir(parents=True)
    for target,source in {
        'scripts/groupmate_judge.py':REFERENCE/'scripts/judge_replies.py',
        'scripts/run_mt5_distill_judge_v1.py':ROOT/'scripts/run_mt5_distill_judge_v1.py',
        'data/evaluation/personas.csv':REFERENCE/'data/evaluation/personas.csv',
        'data/evaluation/judge_rubric.md':REFERENCE/'data/evaluation/judge_rubric.md',
        'reference_judge_run.json':REFERENCE/'experiments/judging/final_current_20261108_01/judge_run.json',
        'source_checkpoint_verification.json':ROOT/'artifacts/mt5_warm_distill_v1/full_results_20261121/checkpoint_verification.json',
        'source_warm_results.json':ROOT/'experiments/mt5_warm_distill_v1_results.json',
        **{f'src/replycoach/{name}':ROOT/f'src/replycoach/{name}' for name in ['__init__.py','io.py','schema.py']},
    }.items():
        dest=STAGE/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    public=[g['public'] for g in packet]
    (STAGE/'judge_inputs.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
    mapping={g['public']['candidate_id']:{**g['private'],'conversation_id':g['public']['conversation_id'],'style':g['public']['style']} for g in packet}
    (STAGE/'PRIVATE_mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
    reference=json.loads((STAGE/'reference_judge_run.json').read_text())
    personas=list(csv.DictReader((STAGE/'data/evaluation/personas.csv').open(encoding='utf-8-sig')))
    assert personas==reference['personas'] and len(personas)==10
    assert (STAGE/'data/evaluation/judge_rubric.md').read_text()==reference['rubric']
    conversations=sorted({r['conversation_id'] for r in public});random.Random(425).shuffle(conversations)
    pilot_ids=conversations[:5]
    pilot_candidates=[r['candidate_id'] for r in public if r['conversation_id'] in pilot_ids]
    config={'experiment':'mt5_distill_judge_v1','remote_root':REMOTE,'models':ARMS,
      'scope':'50 reused development validation conversations; selected checkpoints only; unchanged trained adapter plus four warm-start distillation variants. No final-test inputs.',
      'judge_model':reference['judge_model'],'expected_model_digest':reference['model_digest'],
      'options':reference['options'],'think':reference['think'],'personas':personas,
      'rubric_sha256':sha(STAGE/'data/evaluation/judge_rubric.md'),
      'groupmate_judge_script_sha256':sha(STAGE/'scripts/groupmate_judge.py'),
      'comparison_sha256':sha(comparison),'validation_sha256':sha(ROOT/'data/dev/validation.jsonl'),
      'selected_checkpoint_sources':json.loads((STAGE/'source_checkpoint_verification.json').read_text())['arms'],
      'unique_candidates':len(public),'personas_count':10,'system_style_cells':500,'expanded_ratings':5000,'unique_requests':len(public)*10,
      'workers':4,'shuffle_seed':425,'pilot_candidate_ids':pilot_candidates,'pilot_conversations':pilot_ids,
      'pilot_persona_ids':['P01'],'pilot_unique_requests':len(pilot_candidates),
      'retry_attempts':2,'deduplication':'Only exact reply text within the same conversation and requested style, separately for every persona; retain mapping to all original systems.',
      'engine_difference':'Four concurrent fresh-context requests instead of the prior sequential runner, with unchanged judge_one implementation/model/options/personas/rubric. Hardware/runtime differences limit cross-run numerical comparisons; all five systems are judged together in this new run.',
      'aggregation':'Average ten persona ratings within each conversation/system/style, then average 50 conversations. Paired bootstrap resamples whole conversations; correlated personas are not independent human raters.',
      'pilot_gate':'Exact model identity and protocol; all requested structured responses valid and complete; GPU execution confirmed. Inspect pilot before main launch; no score-based selection or prompt tuning.',
      'paired_meaning':'Single-draft team schema cannot measure English/Singlish consistency; keep existing paired qualitative review separate.',
      'no_training':True,'default_changed':False}
    (STAGE/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    for mode,wall in [('pilot','00:30:00'),('main','12:00:00')]:
        sbatch=f'''#!/bin/bash
#SBATCH --job-name=mt5judge-{mode}
#SBATCH --partition=project
#SBATCH --account=cs425
#SBATCH --qos=cs425qos
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time={wall}
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
cd {REMOTE}
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export OLLAMA_HOST="127.0.0.1:$((22000 + SLURM_JOB_ID % 20000))"
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
export OLLAMA_MODELS="$PWD/model_store"
export OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NOPRUNE=true
export OLLAMA_KEEP_ALIVE=24h
RCS_OLLAMA_BIN="$HOME/.local/share/replycoach-ollama/bin/ollama"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$RCS_OLLAMA_BIN" serve > "ollama-${{SLURM_JOB_ID}}.log" 2>&1 &
RCS_SERVER_PID=$!
trap 'kill "$RCS_SERVER_PID" 2>/dev/null || true' EXIT
ready=0
for attempt in {{1..60}}; do
 if curl --silent --fail "$OLLAMA_BASE_URL/api/tags" >/dev/null; then ready=1; break; fi
 kill -0 "$RCS_SERVER_PID"
 sleep 1
done
test "$ready" = 1
"$HOME/ReplyCoachSG/.venv/bin/python" -u scripts/run_mt5_distill_judge_v1.py --mode {mode}
'''
        (STAGE/f'{mode}.sbatch').write_text(sbatch)
    manifest={str(p.relative_to(STAGE)):sha(p) for p in sorted(STAGE.rglob('*')) if p.is_file()}
    (STAGE/'payload_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(OUT/'payload.tgz','w:gz') as tar:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(STAGE)))
    (OUT/'preparation.json').write_text(json.dumps({'config':config,'payload_sha256':sha(OUT/'payload.tgz'),'files':len(manifest)},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'unique_candidates':len(public),'main_requests':len(public)*10,'expanded_ratings':5000,'pilot_requests':len(pilot_candidates),'files':len(manifest),'payload_sha256':sha(OUT/'payload.tgz')},indent=2))

if __name__=='__main__':main()
