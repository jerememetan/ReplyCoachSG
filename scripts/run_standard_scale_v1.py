"""Frozen, single-variant STANDARD-only scaling experiment."""
import hashlib,json,os,sys
from pathlib import Path
from replycoach.generate.fit_probe_v1 import fit,checked_encoding,generate_standard
from replycoach.generate.finetune_v5 import item_rows

root=Path.cwd()
recipe=json.loads(Path('recipe.json').read_text())
for name,digest in recipe['files_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
variant=sys.argv[1]
method,rate=recipe['variants'][variant]
rows=[r for r in item_rows(Path(recipe['training_file'])) if r['style']=='STANDARD']
assert len(rows)==790 and len({r['item_id'] for r in rows})==790
validation=[r for r in item_rows(Path('data/dev/validation.jsonl')) if r['style']=='STANDARD']
assert not {r['item_id'] for r in rows}&{r['item_id'] for r in validation}
base=Path(os.environ['FOCUS_BASE'])
metadata=dict(variant=variant,method=method,learning_rate=rate,train_items=len(rows),steps=recipe['steps'],seed=425,base=str(base),base_files_sha256={})
for path in sorted(base.glob('*')):
    if path.is_file() and path.suffix in ('.safetensors','.json','.model'):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        metadata['base_files_sha256'][path.name]=h.hexdigest()
Path('artifacts').mkdir(exist_ok=True)
Path(f'artifacts/{variant}_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata),flush=True)
report,model,tokenizer=fit(str(base),rows,root/'artifacts'/variant,method,learning_rate=rate,steps=recipe['steps'],batch_size=4,seed=425,record_steps=(0,recipe['steps']))
assert report['optimizer_steps']==recipe['steps'] and report['train_items']==790
checked_encoding(validation,tokenizer)
report['development_generations']=generate_standard(model,tokenizer,validation,'cuda',4)
report['run_metadata']=metadata
report['limitation']='STANDARD-only experimental checkpoint. Reused development evaluation; no test evaluation, LOCAL retention assessment, or production promotion.'
report['status']='complete'
Path(f'artifacts/{variant}/report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(variant=variant,status='complete')),flush=True)
