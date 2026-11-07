"""Two predeclared learning rates per adaptation method on24 STANDARD examples."""
import gc,hashlib,json,os,sys
from pathlib import Path
import torch
from replycoach.generate.fit_probe_v1 import fit,checked_encoding,generate_standard
from replycoach.generate.train_v5 import validation_loss
ROOT=Path.cwd();BASE=Path(os.environ['FOCUS_BASE']);method=sys.argv[1];assert method in ('lora','full')
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
train_path=ROOT/'data/training_diagnostics/fit_probe_v1/train.jsonl';para_path=ROOT/'data/training_diagnostics/fit_probe_v1/paraphrases.jsonl'
rows=[json.loads(l) for l in train_path.read_text().splitlines()];assert len(rows)==24
metadata=dict(method=method,base=str(BASE),base_files_sha256={},data_sha256={'train':hashlib.sha256(train_path.read_bytes()).hexdigest(),'paraphrases':hashlib.sha256(para_path.read_bytes()).hexdigest()},learning_rates=[1e-4,3e-4],optimizer_updates_each=400,batch_size=4,seed=425)
for path in sorted(BASE.glob('*')):
 if path.is_file() and path.suffix in ('.safetensors','.json','.model'):
  h=hashlib.sha256()
  with path.open('rb') as file:
   for block in iter(lambda:file.read(4*1024*1024),b''):h.update(block)
  metadata['base_files_sha256'][path.name]=h.hexdigest()
(ROOT/'artifacts').mkdir(exist_ok=True);(ROOT/'artifacts'/f'{method}_run_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(dict(method=method,training_examples=24,steps_each=400,learning_rates=[1e-4,3e-4])),flush=True)
for rate,tag in [(1e-4,'lr1e4'),(3e-4,'lr3e4')]:
 name=f'{method}_{tag}';report,model,tokenizer=fit(str(BASE),rows,ROOT/'artifacts'/name,method,learning_rate=rate,steps=400,batch_size=4,seed=425,record_steps=(0,100,400))
 assert report['optimizer_steps']==400 and report['train_items']==24
 # No paraphrase row or loss enters optimization, early stopping or learning-rate selection.
 paraphrases=[json.loads(l) for l in para_path.read_text().splitlines()]
 assert len(paraphrases)==24 and {r['paired_train_id'] for r in paraphrases}=={r['item_id'] for r in rows}
 assert not {r['item_id'] for r in rows}&{r['item_id'] for r in paraphrases}
 encoded=checked_encoding(paraphrases,tokenizer)
 outputs=generate_standard(model,tokenizer,paraphrases,'cuda',4)
 report['paraphrase_generations']=outputs
 report['paraphrase_reference_loss']=validation_loss(model,encoded,4,tokenizer.pad_token_id,'cuda')
 report['paraphrase_exact_matches']=sum(r['exact_target_match'] for r in outputs)
 report['status']='complete';report['base_files_sha256']=metadata['base_files_sha256']
 report['limitation']='STANDARD-only fit diagnostic, not product quality or LOCAL evaluation. One seed,24 underlying scenarios; paraphrases test wording transfer, not independent scenario generalization. Two frozen learning rates reported, not selected post hoc on paraphrase loss.'
 (ROOT/'artifacts'/name/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 (ROOT/'artifacts'/f'{name}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(run=name,steps=400,train_exact_matches=report['history'][-1]['training_exact_matches'],paraphrase_exact_matches=report['paraphrase_exact_matches'],train_loss=report['history'][-1]['teacher_forced_train_loss'],paraphrase_loss=report['paraphrase_reference_loss'])),flush=True)
 del model,tokenizer;gc.collect();torch.cuda.empty_cache()
if method=='lora':
 from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
 model=AutoModelForSeq2SeqLM.from_pretrained(BASE,local_files_only=True).to('cuda').eval();tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 paraphrases=[json.loads(l) for l in para_path.read_text().splitlines()]
 checked_encoding(paraphrases,tokenizer)
 base=dict(method='base',optimizer_steps=0,base_files_sha256=metadata['base_files_sha256'],train_generations=generate_standard(model,tokenizer,rows,'cuda',4),paraphrase_generations=generate_standard(model,tokenizer,paraphrases,'cuda',4),limitation='Unchanged stage1 checkpoint; generated after diagnostic runs, with no optimization or selection.')
 (ROOT/'artifacts/base.json').write_text(json.dumps(base,ensure_ascii=False,indent=2)+'\n');del model,tokenizer;gc.collect();torch.cuda.empty_cache()
