"""Export the two completed candidates using the unchanged portable loader."""
from pathlib import Path
import hashlib,json,os,shutil

ROOT=Path('/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_sealion_large_bc_v1')
DEST=Path.home()/'ReplyCoachSG_sealion_bc548_candidates_v1'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 assert Path.home().name=='hm.teoh.2024'
 recipe=read(ROOT/'recipe.json');original=Path(recipe['original_bundle']);source=read(original/'manifest.json')
 assert sha(original/'manifest.json')==recipe['original_manifest_sha256']
 assert not DEST.exists();DEST.mkdir()
 for n,v in source['files'].items():assert sha(original/n)==v['sha256'],n
 rows=read(ROOT/'frozen_data/evaluation_all.json');ids=['BC-evaluation-00011','BC-evaluation-00037','AUDIT-EVAL-01','AUDIT-EVAL-04','AUDIT-EVAL-08','AUDIT-EVAL-12','USER-FLU-001','ITEM-D-007','ITEM-E-009','ITEM-E-040']
 examples=[{k:r[k] for k in ['item_id','context','speakers','message']} for i in ids for r in rows if r['item_id']==i];assert len(examples)==10
 for arm in ['B','C']:
  dest=DEST/arm;dest.mkdir()
  for name in source['files']:
   if name.startswith(('adapters/','examples/')) or name in ['README.md','training_specs.json']:continue
   src=original/name;out=dest/name;out.parent.mkdir(parents=True,exist_ok=True)
   if name.startswith('base/') and src.suffix=='.safetensors':os.link(src,out)
   else:shutil.copy2(src,out)
  stages={}
  for stage,adapter in [('task','english'),('rewrite','singlish')]:
   report=read(ROOT/'outputs'/arm/stage/'training.json');assert report['status']=='complete'
   selected=(ROOT/'outputs'/arm/stage/'selected').resolve();assert sha(selected/'adapter_model.safetensors')==report['selected_sha256']
   out=dest/'adapters'/adapter;out.mkdir(parents=True)
   for n in ['adapter_config.json','adapter_model.safetensors']:shutil.copy2(selected/n,out/n)
   stages[stage]={k:report[k] for k in ['selected_epoch','selected_sha256','optimizer_updates','total_updates','source_adapter_unchanged','frozen_base_parameters_unchanged']}
  save(dest/'examples/inputs.json',examples)
  results=read(ROOT/'outputs'/arm/'rewrite/selected_pipeline.json');assert results['status']=='complete'
  expected=[{k:r[k] for k in ['item_id','english','singlish','standard_anchor','english_status','singlish_status']} for i in ids for r in results['outputs'] if r['item_id']==i]
  save(dest/'examples/expected_outputs.json',expected)
  save(dest/'training_specs.json',dict(arm=arm,recipe=recipe,stages=stages,originals_unchanged=True,default_changed=False,note='Completed candidate; model-authored targets, small reused development/evaluation sets. Original base weights hard-linked within the same home filesystem; this is not an independent backup. No new token vocabulary or dictionary data added.'))
  (dest/'README.md').write_text(f'''# SEA-LION B/C 548-case candidate {arm}\n\nCompleted four epochs per task and rewrite stage. Selected task epoch{stages['task']['selected_epoch']}, rewrite epoch{stages['rewrite']['selected_epoch']}. Base and tokenizer unchanged; separate English and Singlish adapters. No merging or quantization.\n\nRun: python infer.py --device cuda --inputs examples/inputs.json --output replay.json --verify-examples\n\nUse the existing Python3.13 runtime and included deps/src. Inputs: JSON list with item_id, context, speakers (user/peer/other), message. Verify manifest with python infer.py --verify-only. New output path required for each invocation.\n\nThis candidate has observed contextual and meaning errors; see the project comparison. Ten examples deliberately include known failures and are replay checks, not independent quality evaluation. Keep original model available. No default promotion.\n''')
  files={str(p.relative_to(dest)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(dest.rglob('*')) if p.is_file()}
  save(dest/'manifest.json',dict(kind='sealion',candidate=f'bc548_{arm}',files=files,source_original_manifest_sha256=recipe['original_manifest_sha256'],data_manifest_sha256=sha(ROOT/'frozen_data/manifest.json'),default_changed=False))
  print(arm,str(dest),'manifest_sha256',sha(dest/'manifest.json'),'files',len(files),flush=True)
 for n,v in source['files'].items():assert sha(original/n)==v['sha256'],n
 save(DEST/'export.json',dict(status='exported',candidates={a:str(DEST/a) for a in ['B','C']},original_source_files_verified_unchanged=True))
if __name__=='__main__':main()
