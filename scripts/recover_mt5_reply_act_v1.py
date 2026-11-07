import json
from pathlib import Path
import torch
from replycoach.generate.v5_provider import V5Generator,DECODING
from replycoach.generate.rewrite_v5 import rewrite_standard
root=Path('/common/home/users/h/hm.teoh.2024/ReplyCoachSG_mt5_reply_act_v1')
p=root/'artifacts/results.json';r=json.loads(p.read_text())
rows={}
for split in ['validation']:
 data=[json.loads(l) for l in (root/f'data/dev/{split}.jsonl').read_text().splitlines()]
 rows[split]=[dict(item_id=x['id'],context=[t['text'] for t in x['session_memory']],speakers=[t['speaker'] for t in x['session_memory']],message=x['message']['text'],response='') for x in data]
rows['diagnostic']=[{**x,'response':''} for x in json.loads((root/'data/evaluation/component_baseline_v1/inputs.json').read_text())]
torch.set_num_threads(4)
local=V5Generator(Path(r['rewrite_checkpoint']),device='cuda')
for name,a in r['arms'].items():
 for split,routes in a['outputs'].items():
  for route,items in routes.items():
   for row,item in zip(rows[split],items):
    assert row['item_id']==item['item_id']
    if 'local' in item:continue
    anchor=item['standard']
    if not anchor.strip():
     item['rewrite_status']='blocked_empty_standard';continue
    try:
     text=rewrite_standard(local.model,local.tokenizer,[row],[anchor],local.max_source,local.max_target,'cuda',DECODING,True)[0]
     item.update(local=text,local_standard_anchor=anchor,rewrite_status='complete' if text.strip() else 'failed_empty_local')
    except ValueError as exc:item.update(rewrite_status='failed',rewrite_error=str(exc))
   (root/'artifacts/results_recovered.json').write_text(json.dumps(r,indent=2)+'\n')
   print('Recovered',name,split,route,flush=True)
r['status']='complete_with_failures';r['recovery_note']='Training job failed in rewriting on empty STANDARD oracle output. Original results.json preserved. Recovery only completes postprocessing; blanks remain explicit failures, no retraining.'
(root/'artifacts/results_recovered.json').write_text(json.dumps(r,indent=2)+'\n')
