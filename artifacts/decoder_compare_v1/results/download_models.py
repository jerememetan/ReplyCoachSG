import hashlib,json,os,urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path('/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_decoder_compare_v1')
repo='Qwen/Qwen2.5-3B';revision='3aab1f1954e9cc14eb9509a215f9e5ca08227a9b'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
 return h.hexdigest()
snapshot_download(repo,revision=revision,local_dir=root/'models/qwen',allow_patterns=['*.safetensors','*.json','merges.txt','vocab.json','LICENSE','README.md'],max_workers=2)
with urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'/revision/'+revision+'?blobs=true',timeout=30) as response: metadata=json.load(response)
files={}
for entry in metadata['siblings']:
 p=root/'models/qwen'/entry['rfilename']
 if p.is_file():
  digest=sha(p)
  if entry.get('lfs') and digest!=entry['lfs']['sha256']:raise ValueError('LFS SHA mismatch '+str(p))
  files[str(p)]=digest
base=Path('/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion3b_probe_v1/model')
for p in base.iterdir():
 if p.is_file():files[str(p)]=sha(p)
expected='acba6d6560f61ac72cc95feca5c89f6aabbbf1e9e89c18440237331fcb5ae107'
if files[str(base/'model.safetensors')]!=expected:raise ValueError('SEA-LION base identity changed')
manifest={'status':'complete','qwen_repo':repo,'qwen_revision':revision,'sealion_revision':'836890428766e94322a4b7de27fb081a604703a9','files':files}
(root/'models/download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('DOWNLOAD_VERIFIED',len(files),flush=True)
