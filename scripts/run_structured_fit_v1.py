import hashlib,json,os
from pathlib import Path
from replycoach.generate.structured_fit_v1 import fit,MODES
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
folder=Path('data/training_diagnostics/structured_fit_v1')
def load(name):return [json.loads(l) for l in (folder/f'{name}.jsonl').read_text().splitlines()]
train,permuted,novel=load('train'),load('permuted'),load('novel')
assert (len(train),len(permuted),len(novel))==(12,36,12)
base=Path(os.environ['ACTOR_BASE']);Path('artifacts').mkdir(exist_ok=True)
metadata={}
for p in sorted(base.glob('*')):
    if p.is_file():
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        metadata[p.name]=h.hexdigest()
Path('artifacts/base_sha256.json').write_text(json.dumps(metadata,indent=2)+'\n')
for mode in MODES:
    fit(str(base),train,permuted,novel,Path('artifacts')/mode,mode,**recipe['settings'])
