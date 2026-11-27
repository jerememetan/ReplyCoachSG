from pathlib import Path
import json,hashlib,tarfile
r=Path('/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_sealion_large_bc_v1')
files=[]
for p in sorted(r.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 rel=p.relative_to(r)
 if '__pycache__' in rel.parts or p.suffix in ['.pyc','.pt','.safetensors','.tgz']:continue
 if rel.parts[0]=='outputs' and (any(x.startswith('epoch_') for x in rel.parts[:-1]) or 'selected' in rel.parts):continue
 # All logs/data/code/reports, excluding redundant Ollama server logs.
 if p.name.startswith('ollama-'):continue
 files.append(p)
manifest={str(p.relative_to(r)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
collection=r/'collection548_reports_manifest.json';assert not collection.exists();collection.write_text(json.dumps(dict(files=manifest,source_archive=json.loads((r/'archive_manifest.json').read_text())),indent=2)+'\n')
a=r/'results_only548_v1.tgz';assert not a.exists()
with tarfile.open(a,'w:gz') as t:
 for p in files+[collection]:t.add(p,arcname=str(p.relative_to(r)))
print(json.dumps(dict(archive=str(a),files=len(files)+1,bytes=a.stat().st_size,sha256=hashlib.sha256(a.read_bytes()).hexdigest())))
