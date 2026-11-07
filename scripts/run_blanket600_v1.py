"""Matched control versus all600 user-authorized additions; supervised learning only."""
import gc,hashlib,json,math,os,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from replycoach.generate.finetune_v5 import finetune,item_rows,context_pairs,load_finetuned,validation_outputs
from replycoach.generate.rewrite_v5 import rewrite_rows
from replycoach.generate.style_control import mt5_input
ROOT=Path.cwd();BASE=os.environ['FOCUS_BASE'];DATA=ROOT/'data/training_expansion/blanket600_v1'
arm=sys.argv[1];assert arm in ('control','expanded')
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
manifest=json.loads((DATA/'manifest.json').read_text());assert (manifest['individually_reviewed_additions'],manifest['blanket_authorized_additions'])==(10,590)
tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True);counts={};max_source=0;max_target=0
validation=item_rows(ROOT/'data/dev/validation.jsonl');protected={r['item_id'] for r in validation}
for name,expected in [('control',190),('expanded',790)]:
 path=DATA/f'{name}.jsonl';rows=item_rows(path);inner=context_pairs(path)
 assert len(rows)==expected*2 and not {r['item_id'] for r in rows}&protected
 for r in [*rewrite_rows(rows),*inner]:
  text=mt5_input(r['style'],r['message'],r['context'],True,r['speakers'],standard_reply=r.get('standard_reply'))
  max_source=max(max_source,len(tokenizer(text)['input_ids']));max_target=max(max_target,len(tokenizer(r['response'])['input_ids']))
 counts[name]=dict(conversations=expected,paired_rows=len(rows),context_rows=len(inner),total_rows=len(rows)+len(inner))
source_limit=max(192,math.ceil(max_source/32)*32);target_limit=max(48,math.ceil(max_target/8)*8)
assert source_limit<=512 and target_limit<=128,'Unexpected corpus length; inspect instead of silent truncation'
steps=8*math.ceil(counts['expanded']['total_rows']/16)
preflight=dict(counts=counts,actual_max_source=max_source,actual_max_target=max_target,max_source_tokens=source_limit,max_target_tokens=target_limit,optimizer_steps=steps,epochs_expanded=8,approval_counts={'individual':10,'blanket_authorized':590})
(ROOT/'artifacts').mkdir(exist_ok=True);(ROOT/'artifacts'/f'{arm}_preflight.json').write_text(json.dumps(preflight,indent=2)+'\n');print(json.dumps(preflight),flush=True)
name=f'{arm}_s425'
report=finetune(BASE,train_path=DATA/f'{arm}.jsonl',output_dir=ROOT/'artifacts',output_name=name,seed=425,speaker_labels=True,use_context_pairs=True,rewrite_local=True,epochs=200,patience=100,max_steps=steps,save_steps=(784,),learning_rate=3e-4,batch_size=16,max_source=source_limit,max_target=target_limit,save_final=True)
assert report['optimizer_steps']==steps and report['train_pairs']==counts[arm]['total_rows']
# Diagnostic contexts are evaluation-only and are loaded after training.
diagnostic=item_rows(ROOT/'data/reward_diagnostics/duration_probe_v1.jsonl')
for checkpoint in ['final','step_784']:
 gc.collect();torch.cuda.empty_cache()
 path=ROOT/'artifacts'/name/checkpoint;model=load_finetuned(path,device='cuda').eval();tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 old=validation_outputs(model,tok,validation,source_limit,target_limit,'cuda',speaker_labels=True,rewrite_local=True)
 fresh=validation_outputs(model,tok,diagnostic,source_limit,target_limit,'cuda',speaker_labels=True,rewrite_local=True)
 result=dict(arm=arm,checkpoint=checkpoint,optimizer_steps=steps if checkpoint=='final' else 784,seed=425,validation_outputs=old,diagnostic_outputs=fresh,preflight=preflight,approval_scope='10 individually reviewed,590 blanket-authorized; no new per-item human quality labels or DPO preferences.',limitation='Reused development evaluation; final at matched update budget primary; step784 secondary. Adding conversations also adds valid next-turn context auxiliary rows. Equal steps are not equal FLOPs or original-data exposure.')
 (ROOT/'artifacts'/f'{arm}_{checkpoint}_outputs.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(arm=arm,checkpoint=checkpoint,steps=result['optimizer_steps'],validation_pairs=len(old),diagnostic_pairs=len(fresh))),flush=True)
 del model,tok
