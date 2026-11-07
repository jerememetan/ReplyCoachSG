"""Generate both paired arms on fresh inputs, without training or reference targets."""
import hashlib,json,os
from pathlib import Path
from replycoach.generate.hybrid_v5 import HybridV5Generator
from replycoach.generate.rewrite_v5 import generate_pair
from replycoach.generate.v5_provider import DECODING
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
rows=json.loads(Path('data/human_review/hybrid_pair_ab_v1/inputs.json').read_text());assert len(rows)==30
model=HybridV5Generator(Path(os.environ['ACTOR_BASE']),Path(os.environ['REWRITE_CHECKPOINT']),device='cuda')
report=dict(status='running',inputs=rows,outputs=[],standard_checkpoint=str(model.standard.checkpoint),rewrite_checkpoint=str(model.local.checkpoint),training=False)
Path('artifacts').mkdir(exist_ok=True);output=Path('artifacts/paired_outputs.json')
def save():output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
try:
    for start in range(0,len(rows),4):
        batch=rows[start:start+4];hybrid=model.generate_rows(batch);local=model.local
        earlier=generate_pair(local.model,local.tokenizer,batch,local.max_source,local.max_target,local.device,DECODING,local.format['speaker_labels'])
        for i,(row,new) in enumerate(zip(batch,hybrid)):
            assert new['standard']==new['local_conditioned_on']
            old_standard=earlier['STANDARD'][i].strip();old_local=earlier['LOCAL'][i].strip()
            assert old_standard and old_local
            report['outputs'].append(dict(item_id=row['item_id'],hybrid=dict(standard=new['standard'],local=new['local'],local_conditioned_on=new['standard']),earlier_paired=dict(standard=old_standard,local=old_local,local_conditioned_on=old_standard)))
        save();print('Compared',len(report['outputs']),'/30',flush=True)
    report['status']='complete';save()
except Exception:
    report['status']='failed';save();raise
