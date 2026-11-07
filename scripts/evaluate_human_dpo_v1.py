"""Per-pair held-out evaluation of fixed pre-DPO and selected DPO checkpoints."""
import json,os,gc
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.human_dpo_v1 import encode_preferences
from replycoach.generate.finetune_v5 import sequence_logprobs
folder=Path('data/training_preferences/human_dpo_v1')
rows=[json.loads(l) for l in (folder/'validation.jsonl').read_text().splitlines()]
manifest=json.loads((folder/'manifest.json').read_text())
assert len(rows)==17 and not {r['item_id'] for r in rows}&set(manifest['train_ids'])
base=Path(os.environ['DPO_CHECKPOINT']);best=Path('artifacts/human_dpo/best')
tok=AutoTokenizer.from_pretrained(base,local_files_only=True);encoded=encode_preferences(rows,tok)
results=[]
for path in (base,best):
 model=AutoModelForSeq2SeqLM.from_pretrained(path,local_files_only=True).to('cuda').eval();scores=[]
 with torch.no_grad():
  for start in range(0,len(encoded),4):
   batch=encoded[start:start+4];n=len(batch)
   logs=sequence_logprobs(model,[x[0] for x in batch]+[x[1] for x in batch],tok.pad_token_id,'cuda').cpu().tolist()
   scores.extend([dict(chosen_logp=logs[i],rejected_logp=logs[n+i],margin=logs[i]-logs[n+i]) for i in range(n)])
 results.append(scores);del model;gc.collect();torch.cuda.empty_cache()
output=[]
for row,before,after in zip(rows,*results):
 delta=after['margin']-before['margin']
 output.append(dict(**row,baseline=before,dpo=after,margin_change=delta,dpo_loss=float(-torch.nn.functional.logsigmoid(torch.tensor(.1*delta)))))
Path('artifacts/human_dpo/heldout_pair_evaluation.json').write_text(json.dumps(dict(pairs=output,beta=.1,scope='17 DPO-held-out pairs over8 previously reviewed conversations; not locked test'),ensure_ascii=False,indent=2)+'\n')
print('Evaluated all17 held-out preference pairs',flush=True)
