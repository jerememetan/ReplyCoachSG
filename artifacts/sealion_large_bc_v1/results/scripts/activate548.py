import json,tarfile,hashlib,subprocess
from pathlib import Path
r=Path('/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_sealion_large_bc_v1')
a=r/'payload_subset548.tgz'
assert hashlib.sha256(a.read_bytes()).hexdigest()=='10dc8bef359ff13ad8e7d9e307a028407ca7eaa96f7e54bf886ce1e454871156'
state=subprocess.check_output(['squeue','-h','-j','358870','-o','%T %r'],text=True).strip()
assert state.startswith('PENDING') and 'held' in state.lower(),state
assert not (r/'frozen_data').exists() and not (r/'ready.json').exists()
with tarfile.open(a) as t:
 for m in t.getmembers():
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts,m.name
 t.extractall(r)
recipe=json.loads((r/'recipe.json').read_text())
for p,h in recipe['payload_manifest'].items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
print('VERIFIED_ALL_PAYLOAD_FILES',len(recipe['payload_manifest']))
subprocess.run(['scontrol','release','358870'],check=True)
print('RELEASED_SHARED_PREFLIGHT_358870')
