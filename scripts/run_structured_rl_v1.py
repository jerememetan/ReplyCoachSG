import hashlib,json,os,gc
from pathlib import Path
import torch
from transformers import AutoTokenizer
from replycoach.generate.structured_rl_v1 import encode_cases
from replycoach.generate.structured_rl_train_v1 import fit,load_model,evaluate
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
folder=Path('data/training_diagnostics/structured_rl_v1')
def load(split):return [json.loads(l) for l in (folder/f'{split}.jsonl').read_text().splitlines()]
train,validation,test=load('train'),load('validation'),load('test')
assert (len(train),len(validation),len(test))==(192,48,48)
base=os.environ['ACTOR_BASE'];settings=recipe['settings']
Path('artifacts').mkdir(exist_ok=True)
base_hashes={}
for path in sorted(Path(base).glob('*')):
    if path.is_file():
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        base_hashes[path.name]=h.hexdigest()
Path('artifacts/base_manifest.json').write_text(json.dumps(dict(base=base,files_sha256=base_hashes),indent=2)+'\n')
for arm in ('rl','sft'):
    report=fit(base,train,validation,test,Path('artifacts')/arm,arm,**settings)
    print(json.dumps(dict(arm=arm,status=report['status'],test_accuracy=report['test']['greedy_accuracy'])),flush=True)
model=load_model(base,'cuda');tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
baseline=evaluate(model,test,encode_cases(test,tok),tok.pad_token_id,'cuda')
Path('artifacts/baseline.json').write_text(json.dumps(dict(status='complete',test=baseline),indent=2)+'\n')
print(json.dumps(dict(baseline_test_accuracy=baseline['greedy_accuracy'])),flush=True)
