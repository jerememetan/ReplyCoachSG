import json,hashlib,tarfile,collections
from pathlib import Path
r=Path('/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_sealion_large_bc_v1')
p=r/'before_548_snapshot.tgz'
assert not p.exists()
with tarfile.open(p,'w:gz') as t:
 for n in ['authoring','data','recipe.json','scripts','data.sbatch','B.sbatch','C.sbatch','hardware.json']:
  t.add(r/n,arcname=n)
lines=(r/'authoring/accepted.jsonl').read_bytes().splitlines(keepends=True)
rows=[json.loads(x) for x in lines[:548]]
assert len(rows)==548
b=b''.join(lines[:548]);s=r/'authoring/accepted_first548.jsonl';assert not s.exists();s.write_bytes(b)
print(json.dumps(dict(preserved_total=len(lines),selected=len(rows),counts=dict(collections.Counter(x['split'] for x in rows)),sha256=hashlib.sha256(b).hexdigest(),backup_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
