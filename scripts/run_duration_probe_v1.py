"""Capture a fixed training trajectory, then inspect four frozen step checkpoints."""
import gc,hashlib,json,os,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from replycoach.generate.finetune_v5 import finetune,item_rows,context_pairs,load_finetuned,validation_outputs
from replycoach.generate.rewrite_v5 import rewrite_rows
from replycoach.generate.style_control import mt5_input
from replycoach.generate.train_v5 import encode_rows,validation_loss
ROOT=Path.cwd();BASE=os.environ['FOCUS_BASE'];STEPS=(160,320,480,784)
arm=sys.argv[1];assert arm in ('control','capacity')
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
train=ROOT/'data/synthetic/standard_focus_v1'/('control.jsonl' if arm=='control' else 'targeted.jsonl')
rows=item_rows(train);old=item_rows(ROOT/'data/dev/validation.jsonl');tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
assert len(context_pairs(train))==180
assert len(rows)==(380 if arm=='control' else 540)
for row in rewrite_rows(rows):
 source=mt5_input(row['style'],row['message'],row['context'],True,row['speakers'],standard_reply=row.get('standard_reply'))
 assert len(tokenizer(source)['input_ids'])<=192,row['item_id']
 assert len(tokenizer(row['response'])['input_ids'])<=48,row['item_id']
print('Frozen source/data hashes, unchanged training inputs/context rows and tokenizer limits verified.',flush=True)
args=dict(lora_targets=('q','k','v','o','wi_0','wi_1','wo')) if arm=='capacity' else {}
name=f'{arm}_s425'
report=finetune(BASE,train_path=train,output_dir=ROOT/'artifacts',output_name=name,seed=425,speaker_labels=True,
 use_context_pairs=True,rewrite_local=True,epochs=30,patience=100,max_steps=784,save_steps=STEPS,
 learning_rate=3e-4,batch_size=16,max_source=192,max_target=48,save_final=True,**args)
assert report['optimizer_steps']==784 and set(report['step_checkpoints'])==set(map(str,STEPS))
assert report['train_pairs']==(560 if arm=='control' else 720)
# Fresh cases are not even loaded until all updates and snapshots are complete.
fresh=item_rows(ROOT/'data/reward_diagnostics/duration_probe_v1.jsonl')
assert len(fresh)==48 and not {r['item_id'] for r in fresh}&{r['item_id'] for r in rows+old}
standard_rows=[r for r in old if r['style']=='STANDARD']
for step in STEPS:
 gc.collect();torch.cuda.empty_cache()
 path=ROOT/'artifacts'/name/f'step_{step}'
 model=load_finetuned(path,device='cuda').eval();tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
 loss=validation_loss(model,encode_rows(standard_rows,tokenizer,192,48,speaker_labels=True),32,tokenizer.pad_token_id,'cuda')
 outputs=validation_outputs(model,tokenizer,fresh,192,48,'cuda',speaker_labels=True,rewrite_local=True)
 secondary=validation_outputs(model,tokenizer,old,192,48,'cuda',speaker_labels=True,rewrite_local=True)
 for source,generated in [(fresh,outputs),(old,secondary)]:
  originals={r['item_id']:r for r in source if r['style']=='STANDARD'}
  for r in generated:
   c=originals[r['item_id']]
   for style,anchor in [('STANDARD',None),('LOCAL_REWRITE',r['generated_standard'])]:
    if style=='LOCAL_REWRITE' and not anchor.strip():continue
    text=mt5_input(style,c['message'],c['context'],True,c['speakers'],standard_reply=anchor)
    assert len(tokenizer(text)['input_ids'])<=192,(r['item_id'],style,'generation source truncated')
 if step==784:
  expected={r['item_id']:(r['generated_standard'],r['generated_local']) for r in report['final_validation_outputs']}
  assert all((r['generated_standard'],r['generated_local'])==expected[r['item_id']] for r in secondary),'Reloaded step784 differs from final weights'
 result=dict(arm=arm,seed=425,optimizer_step=step,checkpoint=str(path),original_reference_standard_loss=loss,
  fresh_source_sha256=recipe['files_sha256']['data/reward_diagnostics/duration_probe_v1.jsonl'],outputs=outputs,
  secondary_validation_outputs=secondary,limitation='Fresh assistant-authored development probe, not human gold. Original50 generation retained as secondary; token loss is not reply accuracy. References/guidance excluded from generation input.')
 (ROOT/'artifacts'/f'{arm}_step{step}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(arm=arm,step=step,standard_reference_loss=loss,fresh_pairs=len(outputs),secondary_pairs=len(secondary))),flush=True)
 del model,tokenizer
