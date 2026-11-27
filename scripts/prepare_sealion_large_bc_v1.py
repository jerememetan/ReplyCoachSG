"""Package only the authorized B and C arms and a shared authoring dependency."""
import hashlib,json,random,shutil,tarfile
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/sealion_large_bc_v1'
REMOTE='/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_sealion_large_bc_v1'
ASSETS='/common/scratch/users/h/hm.teoh.2024/replycoach_public_assets_bc_v1'
PYTHON='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def main():
 stage=OUT/'stage';assert not stage.exists();stage.mkdir(parents=True)
 old=ROOT/'artifacts/sealion_audited_restart_v1/stage'
 shutil.copytree(old/'data',stage/'data')
 shutil.copytree(ROOT/'src',stage/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ['protected_train.jsonl','protected_validation.jsonl']:shutil.copy2(old/name,stage/name)
 (stage/'scripts').mkdir();(stage/'tests').mkdir()
 for name in ['run_sealion_large_bc_v1.py','sealion_large_bc_data.py']:shutil.copy2(ROOT/'scripts'/name,stage/'scripts'/name)
 for name in ['test_sealion_large_bc_v1.py','test_sealion_audited_restart_v1.py']:shutil.copy2(ROOT/'tests'/name,stage/'tests'/name)
 spec=importlib.util.spec_from_file_location('authoring',ROOT/'scripts/sealion_large_bc_data.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 slots=[]
 for split,count in [('train',5000),('dev',120),('evaluation',120)]:
  for i in range(count):
   kind=m.KINDS[((i//10+i%10)%10) if i%10<7 else (10+(i//10)%4)]
   slots.append(dict(slot=f'{split}-{i:05d}',split=split,topic=m.TOPICS[split][i%len(m.TOPICS[split])],kind=kind,
       input_register=['short_text_english','casual_english','light_singlish','short_text_english','light_singlish'][i%5],
       history_turns=i%5,history_pattern=([['peer','user','peer','user'][:i%5],['user','peer','user','peer'][:i%5],['user','user','peer','user'][:i%5]][(i//5)%3]),draft_relation='different valid choice, commitment or reply act if constraints allow' if i%2==0 else 'same grounded answer with different wording or a grounded additional clause'))
 # Generate early samples from all strata, then interleave dev/eval with training.
 # Freeze memberships before any model authoring or student training.
 rng=random.Random(425);rng.shuffle(slots)
 # First sample uses train only; final evaluation inputs are not exposed for early assistant sampling.
 first=[s for s in slots if s['split']=='train'][:30];ids={s['slot'] for s in first};slots=first+[s for s in slots if s['slot'] not in ids]
 save(stage/'data/authoring_plan.json',dict(seed=425,slots=slots,max_batches=3500,requested_counts=dict(train=5000,dev=120,evaluation=120),topic_membership=m.TOPICS,coverage='70% everyday and 30% grounded fact categories by requested slot; semantic fit is model-reviewed, not human-certified.'))
 recipe=json.loads((old/'recipe.json').read_text())
 recipe.update(experiment='sealion_large_bc_v1',remote_root=REMOTE,arms={
   'B':dict(task=dict(learning_rate=1e-4,max_epochs=4),rewrite=dict(learning_rate=3e-5,max_epochs=4)),
   'C':dict(task=dict(learning_rate=5e-5,max_epochs=4),rewrite=dict(learning_rate=1e-5,max_epochs=4))},
   scheduler=dict(type='linear warmup then cosine',warmup_fraction=.05,end_lr=0.,step_unit='optimizer update',total_updates='ceil(actual encoded examples / 8) * 4',first_update='peak_lr / warmup_updates'),
   data=dict(new_conversations=5000,retained_audited_conversations=768,total_task_conversations=5768,task_targets=11536,new_rewrite_pairs=10000,retained_rewrite_pairs=766,total_rewrite_pairs=10766,new_dev_conversations=120,new_final_conversations=120),
   budget=dict(authoring_checkpoint_after_hours=20,authoring_requeues_same_job_id=True,epochs_per_stage=4,early_stopping=False,save_every_epoch=True,epoch0_eligible=True),
   selection='Provisional: 0.5 old reference CE + 0.5 new dev reference CE. Save all epochs and dev generations. Generation review before any promotion; CE is not semantic accuracy.',
   generation=dict(writer=m.WRITER,critic=m.CRITIC,offline=True,source='Existing cached public model assets copied to personal scratch; no cloud calls',raw_logs=True,assistant_initial_sample_gate=24,all_new_targets_model_reviewed=True,additional_semantic_review=m.WRITER,semantic_review_thinking=True,review_format="Fact/error fields before a final accept/reject verdict",review_calibration="Gemma and combined gate must match all known good/bad fixtures; Qwen alone is not trusted",review_note="Separate critic plus a fresh author-model semantic pass; neither is human gold",human_gold=False),
   gpu=dict(allowed_features=['l40s','rtx_pro_6000_blackwell_max'],allocation='one untyped GPU with Slurm OR constraint; never another architecture',concurrent_training_arms=2),
   validation=dict(old=50,old_rewrite=42,new=120,new_rewrite=240,topic_scope='Topic partition applies to new model-authored examples only; historical and public overlap possible'),
   evaluation=dict(new=120,reused=75,total=195,uses_final_for_selection=False,new_fixed_reference_rewrites=120),
   comparison='Only B/C are trained. Same frozen data, seed, public initialization, adapter structure, inference and full budget. Peak rates differ jointly across stages; cannot isolate which stage caused a difference. Hardware may differ; actual device logged.',
   limitations='Synthetic labels and a separate model critic can still share systematic errors. Five user preference samples calibrate style, not all new labels. Final cases become development material once examined. No claim that scheduler fixes the problem. Full checkpoints and default preserved.',
   default_changed=False)
 # Remove obsolete counts/old constant-rate stage recipe.
 recipe.pop('task',None);recipe.pop('rewrite',None);recipe.pop('payload_manifest',None)
 prefix=['#!/bin/bash','#SBATCH --partition=student','#SBATCH --account=cs701','#SBATCH --qos=studentqos','#SBATCH --cpus-per-task=4','#SBATCH --mem=32G','#SBATCH --time=24:00:00','#SBATCH --requeue','#SBATCH --open-mode=append','#SBATCH --gres=gpu:1','#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max','#SBATCH --output=slurm-%x-%j.out','set -euo pipefail','module purge','module load Python/3.13.1-GCCcore-13.3.0',f'cd {REMOTE}',f'export PYTHONPATH="$PWD/src:{recipe["original_bundle"]}/deps" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false']
 prep=prefix[:1]+['#SBATCH --job-name=sea-bc-data']+prefix[1:]+[
  f'{PYTHON} -u scripts/run_sealion_large_bc_v1.py --stage hardware',
  f'export PATH="{ASSETS}/.local/share/replycoach-ollama/bin:$PATH"',
  f'export OLLAMA_MODELS="{ASSETS}/.cache/replycoach-ollama/models"',
  'export OLLAMA_HOST="127.0.0.1:$((20000+SLURM_JOB_ID%30000))" OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=2',
  'export OLLAMA_BASE_URL="http://$OLLAMA_HOST"',
  'ollama serve > ollama-${SLURM_JOB_ID}.log 2>&1 &',
  'server_pid=$!',"trap 'kill \"$server_pid\" 2>/dev/null || true' EXIT",
  'for i in $(seq 1 90); do if curl -fsS "$OLLAMA_BASE_URL/api/tags" > installed_models.json; then break; fi; sleep 2; done',
  f'if {PYTHON} -u scripts/sealion_large_bc_data.py; then',
  '  :',
  'else',
  '  authoring_status=$?',
  '  if [ "$authoring_status" -eq 75 ]; then',
  '    scontrol requeue "$SLURM_JOB_ID"',
  '    exit 0',
  '  fi',
  '  exit "$authoring_status"',
  'fi',
  'kill "$server_pid"; wait "$server_pid" || true',
  'trap - EXIT',
  f'{PYTHON} -u scripts/run_sealion_large_bc_v1.py --stage preflight','']
 # Quote shell variable literally without expanding anything during packaging.
 for name,lines in [('data',prep)]+[(arm,prefix[:1]+[f'#SBATCH --job-name=sea-bc-{arm}']+prefix[1:]+[f'{PYTHON} -u scripts/run_sealion_large_bc_v1.py --stage pipeline --arm {arm}','']) for arm in ['B','C']]:
  (stage/f'{name}.sbatch').write_text('\n'.join(lines))
 recipe['payload_manifest']={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
 save(stage/'recipe.json',recipe);save(OUT/'recipe.json',recipe);save(ROOT/'experiments/sealion_large_bc_v1_recipe.json',recipe)
 with tarfile.open(OUT/'payload.tgz','w:gz') as tar:
  for p in sorted(stage.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(stage)),recursive=False)
 save(OUT/'package.json',dict(payload_sha256=sha(OUT/'payload.tgz'),bytes=(OUT/'payload.tgz').stat().st_size,files=len(recipe['payload_manifest'])+1))
 print(json.dumps(json.loads((OUT/'package.json').read_text())))
if __name__=='__main__':main()
