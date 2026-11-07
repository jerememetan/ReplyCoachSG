import json,hashlib,os
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.fit_probe_v1 import generate_standard
recipe=json.loads(Path('recipe.json').read_text())
for f,d in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==d,f
rows=json.loads(Path('fresh_eval.json').read_text())
base=os.environ['ACTOR_BASE'];model=AutoModelForSeq2SeqLM.from_pretrained(base,local_files_only=True).to('cuda').eval();tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
outputs=generate_standard(model,tok,rows,'cuda',4)
for r,g in zip(rows,outputs):r['standard']=g['generated_standard'];assert r['standard'].strip()
Path('anchors.json').write_text(json.dumps(dict(status='complete',checkpoint=base,items=rows),ensure_ascii=False,indent=2)+'\n')
print('Generated40 fixed STANDARD anchors',flush=True)
