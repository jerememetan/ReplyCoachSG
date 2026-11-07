"""Existing mT5 STANDARD and LOCAL components on matched new diagnostic cases."""
import gc,json,os
from pathlib import Path
import torch
from replycoach.generate.v5_provider import V5Generator,DECODING
from replycoach.generate.train_v5 import generate_replies
from replycoach.generate.rewrite_v5 import rewrite_standard
from replycoach.generate.native_rewrite_v2 import source
from replycoach.generate.style_control import mt5_input
from run_component_baseline_v1 import FOLDER,verify_recipe

def check_budget(rows,tok,task,limit,standard=None):
 for n,r in enumerate(rows):
  text=mt5_input(task,r['message'],r['context'],True,r['speakers'],**({'standard_reply':standard[n]} if standard else {}))
  if len(tok(text)['input_ids'])>limit:raise ValueError('mT5 input would be truncated')
def inference_rows(rows):
 return [{**r,"response":""} for r in rows]

def main():
 verify_recipe();rows=inference_rows(json.loads((FOLDER/'inputs.json').read_text()));refs={r['item_id']:r for r in json.loads((FOLDER/'reference_anchors.json').read_text())}
 standard_path=Path('/common/home/users/h/hm.teoh.2024/ReplyCoachSG_standard_val_v1/artifacts/full_lr3e4/best')
 local_path=Path('/common/home/users/h/hm.teoh.2024/ReplyCoachSG_hybrid_inference_v1/checkpoints/local_rewrite')
 model=V5Generator(standard_path,device='cuda');check_budget(rows,model.tokenizer,'STANDARD',model.max_source)
 generated=generate_replies(model.model,model.tokenizer,rows,'STANDARD',model.max_source,model.max_target,'cuda',DECODING,True)
 report=dict(status='running',standard_checkpoint=str(standard_path),rewrite_checkpoint=str(local_path),cases=[dict(item_id=r['item_id'],direct_standard=s) for r,s in zip(rows,generated)],limitations=['Existing task-adapted mT5 versus prompted chat models is a practical system comparison, not isolated size or instruction-tuning effects.','mT5 does not have a trained explicit-plan task; no untrained planning prompt is used as a capability verdict.'])
 out=Path('artifacts');out.mkdir(exist_ok=True);path=out/'mt5.json'
 path.write_text(json.dumps(report,indent=2)+'\n');del model;gc.collect();torch.cuda.empty_cache()
 local=V5Generator(local_path,device='cuda')
 for route,anchors in [('direct',generated),('oracle',[refs[r['item_id']]['standard'] for r in rows])]:
  check_budget(rows,local.tokenizer,'LOCAL_REWRITE',local.max_source,anchors)
  outputs=rewrite_standard(local.model,local.tokenizer,rows,anchors,local.max_source,local.max_target,'cuda',DECODING,True)
  for c,a,s in zip(report['cases'],anchors,outputs):c[route+'_local']=dict(standard_anchor=a,reply=s)
  path.write_text(json.dumps(report,indent=2)+'\n')
 report['status']='complete';path.write_text(json.dumps(report,indent=2)+'\n');print('Completed24 matched mT5 pairs and24 oracle rewrites',flush=True)
if __name__=='__main__':main()
