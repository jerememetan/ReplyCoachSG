"""Freeze an audited768-only B/C comparison with full epochs and warmup/cosine."""
from pathlib import Path
import hashlib,json,shutil,tarfile,datetime
ROOT=Path(__file__).resolve().parents[1]
NAME='sealion_audited768_bc_warmup_v1'
OUT=ROOT/'artifacts'/NAME
REMOTE='/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_'+NAME
OLD=ROOT/'artifacts/sealion_large_bc_v1/results'
PYTHON='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 assert not OUT.exists();stage=OUT/'stage';stage.mkdir(parents=True)
 original_recipe=read(OLD/'recipe.json')
 for n in ['task_train.json','rewrite_train.json','old_validation.json','old_rewrite_validation.json','style_validation.json','evaluation.json']:
  assert sha(OLD/'data'/n)==original_recipe['payload_manifest']['data/'+n],n
 for n,h in original_recipe['source_protected_hashes'].items():assert sha(OLD/n)==h,n
 for n,h in [('data/dev/train.jsonl','03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7'),('data/dev/validation.jsonl','68ab3fcfd283bddd37b8f4ac93a79edcc20a45b1ab9d12ce9732e26f74b20a52')]:assert sha(ROOT/n)==h,n
 shutil.copytree(OLD/'src',stage/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 (stage/'scripts').mkdir();(stage/'tests').mkdir();(stage/'data').mkdir()
 for n in ['test_sealion_large_bc_v1.py','test_sealion_audited_restart_v1.py']:shutil.copy2(OLD/'tests'/n,stage/'tests'/n)
 for n in ['protected_train.jsonl','protected_validation.jsonl']:shutil.copy2(OLD/n,stage/n)
 for n in ['task_train.json','rewrite_train.json','old_validation.json','old_rewrite_validation.json','style_validation.json','evaluation.json','exclusions.json']:shutil.copy2(OLD/'data'/n,stage/'data'/n)
 train=read(stage/'data/task_train.json');rw=read(stage/'data/rewrite_train.json');dev=read(stage/'data/style_validation.json');evaluation=read(stage/'data/evaluation.json')
 assert len(train)==768 and len(rw)==766 and len(dev)==20 and len(evaluation)==75
 frozen=stage/'frozen_data';frozen.mkdir()
 save(frozen/'train_task.json',train);save(frozen/'train_rewrite.json',rw)
 for kind in ['task','rewrite']:save(frozen/f'dev_{kind}.json',dev)
 eval24=[dict(x,scope='reused_assistant_evaluation',previous_scope=x['scope']) for x in evaluation if x['scope']=='fresh_assistant_evaluation'];assert len(eval24)==24
 for kind in ['task','rewrite']:save(frozen/f'evaluation_{kind}.json',eval24)
 save(frozen/'evaluation_all.json',eval24+[x for x in evaluation if x['scope']!='fresh_assistant_evaluation'])
 provenance=dict(status='verified_unchanged_768',source_experiment='sealion_large_bc_v1/data retained from sealion_audited_restart_v1',hashes={n:sha(stage/'data'/n) for n in ['task_train.json','rewrite_train.json','style_validation.json','evaluation.json']},train=768,rewrite=766,new523_included=False,labels='Existing audited/assistant-revised mixed-source labels, not fully human gold.',exclusions=read(stage/'data/exclusions.json'),source_protected_hashes=original_recipe['source_protected_hashes'])
 save(stage/'source_dataset.json',provenance)
 save(frozen/'manifest.json',dict(status='complete',hashes={p.name:sha(p) for p in frozen.glob('*.json')},counts=dict(train=768,dev=20,evaluation=24),source_dataset_sha256=sha(stage/'source_dataset.json'),human_gold=False,split_note='Unchanged768old audited training,20prior assistant development cases,24prior assistant evaluation plus1user/50oldval diagnostic cases. Reused development/evaluation, not independent heldout quality.'))
 code=(ROOT/'scripts/run_sealion_large_bc_v1.py').read_text()
 code=code.replace("assert read('authoring/assistant_sample_review.json')['proceed'] is True", "assert read('source_dataset.json')['status']=='verified_unchanged_768'\n    assert manifest['source_dataset_sha256']==sha(ROOT/'source_dataset.json')\n    for name,h in read('source_dataset.json')['hashes'].items():assert sha(ROOT/'data'/name)==h,name")
 code=code.replace("'new_model_authored_evaluation'","'reused_assistant_evaluation'")
 (stage/'scripts/run_sealion_large_bc_v1.py').write_text(code)
 archive=(ROOT/'scripts/archive_sealion_large_bc_v1.py').read_text().replace("ReplyCoachSG_sealion_large_bc_v1_results","ReplyCoachSG_"+NAME+"_results")
 (stage/'scripts/archive_sealion_large_bc_v1.py').write_text(archive)
 r={k:v for k,v in original_recipe.items() if k not in ['payload_manifest','scope_amendment','generation']}
 r.update(experiment=NAME,remote_root=REMOTE,data=dict(new_conversations=0,retained_audited_conversations=768,total_task_conversations=768,task_targets=1536,new_rewrite_pairs=0,retained_rewrite_pairs=766,total_rewrite_pairs=766,new_dev_conversations=20,new_final_conversations=24),
  validation=dict(old=50,old_rewrite=42,new=20,new_rewrite=20,note='new means historical assistant style-development group, reused'),evaluation=dict(new=24,reused=51,total=75,new_fixed_reference_rewrites=24,uses_final_for_selection=False,note='All cases previously inspected; development diagnostics, not a fresh independent benchmark.'),
  budget=dict(epochs_per_stage=8,early_stopping=False,save_every_epoch=True,epoch0_eligible=True),
  generation=dict(authoring=False,additional_generation=False,all_training_data_preexisting=True),
  scheduler=dict(type='linear warmup then cosine',warmup_fraction=.05,end_lr=0.,step_unit='optimizer update',total_updates='ceil(actual encoded examples /8)*8',first_update='peak_lr/warmup_updates'),
  comparison='Two new B/C pipelines using ONLY the existing768audited conversations, not the completed1291case mixture. Same public initialization and peak LR pairs as priorB/C. Epoch count changes4to8 and validation20replaces11, so this is NOT an isolated scheduler or data ablation.',
  limitations='One seed, reused assistant-reviewed/synthetic sources and extensively reused development/evaluation. Lower referenceCE does not prove fidelity or naturalness; no default promotion.',
  request=dict(user='can u do the700+dataset run with epochs and cooldown/warmup for this sealion?',assumption='Existing audited768case set; clarification offered asynchronously, this is the announced default.',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 for a in ['B','C']:
  for s in ['task','rewrite']:r['arms'][a][s]['max_epochs']=8
 prefix=['#!/bin/bash','#SBATCH --partition=student','#SBATCH --account=cs701','#SBATCH --qos=studentqos','#SBATCH --cpus-per-task=4','#SBATCH --mem=32G','#SBATCH --time=12:00:00','#SBATCH --open-mode=append','#SBATCH --gres=gpu:1','#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max','#SBATCH --output=slurm-%x-%j.out','set -euo pipefail','module purge','module load Python/3.13.1-GCCcore-13.3.0',f'cd {REMOTE}',f'export PYTHONPATH="$PWD/src:{r["original_bundle"]}/deps" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false']
 for name,commands in [('preflight',[f'{PYTHON} -u scripts/run_sealion_large_bc_v1.py --stage hardware',f'{PYTHON} -u scripts/run_sealion_large_bc_v1.py --stage preflight'])]+[(a,[f'{PYTHON} -u scripts/run_sealion_large_bc_v1.py --stage pipeline --arm {a}']) for a in ['B','C']]:
  (stage/(name+'.sbatch')).write_text('\n'.join(prefix[:1]+[f'#SBATCH --job-name=sea768-{name}']+prefix[1:]+commands+['']))
 (stage/'archive.sbatch').write_text('\n'.join(['#!/bin/bash','#SBATCH --job-name=sea768-archive','#SBATCH --partition=student','#SBATCH --account=cs701','#SBATCH --qos=studentqos','#SBATCH --cpus-per-task=1','#SBATCH --mem=4G','#SBATCH --time=00:30:00','#SBATCH --output=slurm-%x-%j.out','set -euo pipefail',f'cd {REMOTE}',f'python3 scripts/archive_sealion_large_bc_v1.py','']))
 r['payload_manifest']={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
 save(stage/'recipe.json',r);save(OUT/'recipe.json',r);save(ROOT/'experiments'/(NAME+'_recipe.json'),r)
 with tarfile.open(OUT/'payload.tgz','w:gz') as t:
  for p in sorted(stage.rglob('*')):
   if p.is_file():t.add(p,arcname=str(p.relative_to(stage)),recursive=False)
 package=dict(payload_sha256=sha(OUT/'payload.tgz'),bytes=(OUT/'payload.tgz').stat().st_size,files=len(r['payload_manifest'])+1)
 save(OUT/'package.json',package);print(json.dumps(package))
if __name__=='__main__':main()
