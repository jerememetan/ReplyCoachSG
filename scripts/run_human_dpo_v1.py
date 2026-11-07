import hashlib,json,os
from pathlib import Path
from replycoach.generate.human_dpo_v1 import fit
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
folder=Path('data/training_preferences/human_dpo_v1')
def load(name):return [json.loads(l) for l in (folder/name).read_text().splitlines() if l.strip()]
train=load('train.jsonl');val=load('validation.jsonl')
manifest=json.loads((folder/'manifest.json').read_text())
assert len(train)==manifest['train_pairs'] and len(val)==manifest['validation_pairs']
audit=[dict(item_id=i['id'],style='STANDARD',context=[t['text'] for t in i['session_memory']],speakers=[t['speaker'] for t in i['session_memory']],message=i['message']['text'],response=i['replies']['standard']) for i in load('audit_conversations.jsonl')]
audit+=load('fresh_audit.jsonl')
base=Path(os.environ['DPO_BASE'])
base_hashes={}
for p in sorted(base.glob('*')):
    if p.is_file():
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        base_hashes[p.name]=h.hexdigest()
Path('base_sha256.json').write_text(json.dumps(base_hashes,indent=2)+'\n')
report=fit(str(base),train,val,audit,Path('artifacts/human_dpo'),**recipe['settings'])
print(json.dumps(dict(status=report['status'],best_epoch=report['best_epoch'])),flush=True)
