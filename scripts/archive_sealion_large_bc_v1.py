"""Archive new experiment evidence and all epoch adapters after both arms finish."""
import hashlib,json,os,tarfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
DEST=Path.home()/'ReplyCoachSG_sealion_large_bc_v1_results'
assert Path.home().name=='hm.teoh.2024'
assert not DEST.exists(),'Existing result archive must not be overwritten'
DEST.mkdir()
stages={}
for arm in ['B','C']:
 for stage in ['task','rewrite']:
  p=ROOT/'outputs'/arm/stage/'training.json'
  stages[arm+'_'+stage]=json.loads(p.read_text()) if p.exists() else {'status':'missing'}
summary=dict(at_utc=datetime.now(timezone.utc).isoformat(),stages={k:{n:v.get(n) for n in ['status','selected_epoch','selected_sha256','optimizer_updates','total_updates','gpu','source_adapter_unchanged','frozen_base_parameters_unchanged']} for k,v in stages.items()},default_changed=False)
(ROOT/'archive_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
partial=DEST/'experiment.tgz.partial'
with tarfile.open(partial,'w:gz',compresslevel=1) as archive:
 for p in sorted(ROOT.rglob('*')):
  # Large optimizer state is redundant for review; preserve all adapter epochs.
  if p.name=='last_optimizer.pt' or '__pycache__' in p.parts or p.suffix=='.pyc':continue
  if p.is_file() or p.is_symlink():archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
final=DEST/'experiment.tgz';partial.replace(final)
h=hashlib.sha256()
with final.open('rb') as f:
 for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
manifest=dict(archive=str(final),bytes=final.stat().st_size,sha256=h.hexdigest(),all_saved_adapter_epochs_included=True,optimizer_states_excluded=True,summary=summary)
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest),flush=True)
