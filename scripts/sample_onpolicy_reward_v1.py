import hashlib,json,os
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.style_control import mt5_input
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
rows=[json.loads(l) for l in Path('data/reward_diagnostics/onpolicy_v1/scenarios.jsonl').read_text().splitlines()]
base=Path(os.environ['ACTOR_BASE']);tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
model=AutoModelForSeq2SeqLM.from_pretrained(base,local_files_only=True).to('cuda').eval()
torch.manual_seed(425)
output=Path('artifacts/samples.json');output.parent.mkdir(exist_ok=True)
if output.exists():raise FileExistsError('Do not replace sampled evidence')
report=dict(status='running',seed=425,checkpoint=str(base),sampling=recipe['sampling'],results=[])
with torch.no_grad():
 for row in rows:
  text=mt5_input('STANDARD',row['message'],row['context'],True,row['speakers'])
  encoded=tok(text,return_tensors='pt');assert encoded['input_ids'].shape[1]<=192
  generated=model.generate(**{k:v.cuda() for k,v in encoded.items()},**recipe['sampling'])
  for n,ids in enumerate(generated):
   ids=ids.tolist()
   report['results'].append(dict(**row,id=f"{row['item_id']}-S{n+1}",draft=tok.decode(ids,skip_special_tokens=True).strip(),ended_with_eos=tok.eos_token_id in ids))
  output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('Sampled',len(report['results']),'/48',flush=True)
report['status']='complete';output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
