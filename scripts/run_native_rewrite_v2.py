import gc,hashlib,json,os,sys
from pathlib import Path
import torch
from replycoach.generate.native_rewrite_train_v2 import fit
recipe=json.loads(Path('recipe.json').read_text())
for f,d in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==d,f
mode=sys.argv[1];assert mode in ('full_context','anchor_only','exemplar')
folder=Path('data/public/native_rewrite_v2')
def load(name):return [json.loads(l) for l in (folder/f'{name}.jsonl').read_text().splitlines()]
train,val=load('train'),load('validation');manifest=json.loads((folder/'manifest.json').read_text())
assert len(train)==manifest['train'] and len(val)==manifest['validation']
anchors=json.loads((folder/'fixed_anchors.json').read_text());assert anchors['status']=='complete'
anchors=anchors['items'];assert len(anchors)==40
base=Path(os.environ['PROJECT_ROOT'])/'artifacts/mt5_base_after_spans_spk/best'
h=hashlib.sha256()
with (base/'model.safetensors').open('rb') as f:
 for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
assert h.hexdigest()==recipe['base_sha256']
Path('artifacts').mkdir(exist_ok=True)
# Predeclared cheap controls on development only, never used to tune the new model.
if mode=='full_context':
 from peft import PeftModel
 from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
 from replycoach.generate.rewrite_v5 import rewrite_standard
 controls=[];development=[r for r in anchors if r['split']=='development']
 checkpoint_root=Path(os.environ['PROJECT_ROOT'])/'experiments/shared_intent_v1/artifacts/rewrite_s425'
 for checkpoint in ('best','final'):
  path=checkpoint_root/checkpoint
  model=AutoModelForSeq2SeqLM.from_pretrained(base,local_files_only=True).to('cuda')
  model=PeftModel.from_pretrained(model,path).eval();tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
  for beams in (1,4):
   outputs=[]
   for n in range(0,len(development),4):
    rows=development[n:n+4];replies=rewrite_standard(model,tok,rows,[r['standard'] for r in rows],192,64,'cuda',dict(num_beams=beams,no_repeat_ngram_size=3),True)
    outputs.extend(dict(item_id=r['item_id'],standard=r['standard'],local=t,local_conditioned_on=r['standard']) for r,t in zip(rows,replies))
   controls.append(dict(checkpoint=checkpoint,num_beams=beams,outputs=outputs))
  del model,tok;gc.collect();torch.cuda.empty_cache()
 Path('artifacts/incumbent_development_controls.json').write_text(json.dumps(controls,ensure_ascii=False,indent=2)+'\n')
report=fit(str(base),train,val,anchors,Path('artifacts')/mode,mode,**recipe['settings'])
print(json.dumps(dict(mode=mode,status=report['status'],best_step=report['best_step'],best_validation_macro=report['best_validation_macro'])),flush=True)
