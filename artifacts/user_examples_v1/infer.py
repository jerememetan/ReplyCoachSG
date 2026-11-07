import json,os,hashlib
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.fit_probe_v1 import generate_standard
rows=json.loads(Path('user_examples.json').read_text())
base=Path(os.environ['ACTOR_BASE'])
tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
model=AutoModelForSeq2SeqLM.from_pretrained(base,local_files_only=True).to('cuda').eval()
outputs=generate_standard(model,tok,rows,'cuda',batch_size=3)
report=dict(checkpoint=str(base),scope='User-supplied examples, STANDARD only, no optimization',decoding=dict(num_beams=4,no_repeat_ngram_size=3,max_new_tokens=48),inputs=rows,outputs=[dict(item_id=r['item_id'],reply=r['generated_standard']) for r in outputs])
Path('outputs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report['outputs'],ensure_ascii=False),flush=True)
