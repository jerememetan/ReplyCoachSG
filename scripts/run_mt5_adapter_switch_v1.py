"""Two independent mT5 task adapters sharing the frozen fully adapted base."""
import gc,hashlib,json,random,traceback
from pathlib import Path
import torch,transformers,peft
from transformers import AutoTokenizer,MT5Config
from peft import PeftModel
from replycoach.generate.mt5_pretrained_compat import OriginalMT5
from replycoach.generate.mt5_matched_v1 import task_rows,paired_rows,encode,source,reply_loss
from replycoach.generate.train_v5 import collate
from replycoach.generate.mt5_adapter_switch_v1 import new_task_adapter,activate_inference,load_switching_adapters
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';R={};TOK=None
DECODE={'greedy':dict(num_beams=1,do_sample=False,no_repeat_ngram_size=0),
        'beam4':dict(num_beams=4,do_sample=False,no_repeat_ngram_size=3)}
def read(p):return json.loads((ROOT/p).read_text())
def rows(p):return [json.loads(l) for l in (ROOT/p).read_text().splitlines() if l.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save():
 OUT.mkdir(exist_ok=True);p=OUT/'results.tmp';p.write_text(json.dumps(R,indent=2)+'\n');p.replace(OUT/'results.json')
def clear():gc.collect();torch.cuda.empty_cache()
def load_base(path):
 raw=json.loads((path/'config.json').read_text());cfg=MT5Config(**raw);cfg.tie_word_embeddings=raw['tie_word_embeddings']
 # The checkpoint stores an independent lm_head even though old HF metadata saysTrue.
 # Preserve that actual tensor mapping with the existing input-sharing-only loader.
 model,info=OriginalMT5.from_pretrained(path,config=cfg,dtype=torch.float32,local_files_only=True,output_loading_info=True)
 problems={k:list(info.get(k) or []) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']}
 if any(problems.values()):raise ValueError('Strict adapted mT5 load failed '+str(problems))
 if not (model.shared.weight is model.encoder.embed_tokens.weight is model.decoder.embed_tokens.weight):raise ValueError('Input-sharing changed')
 if model.lm_head.weight is model.shared.weight:raise ValueError('Stored independent output head lost')
 if sum(p.numel() for p in model.parameters())!=582401280:raise ValueError('Adapted base parameter count changed')
 return model.cuda()
def tensor(batch):
 x,a,y=collate(batch,TOK.pad_token_id);return x.cuda(),a.cuda(),y.cuda()
@torch.no_grad()
def validation(model,examples):
 model.eval();total=0.;count=0
 for i in range(0,len(examples),2):
  x,a,y=tensor(examples[i:i+2]);n=int(y.ne(-100).sum());total+=float(reply_loss(model(input_ids=x,attention_mask=a,labels=y,use_cache=False).logits,y))*n;count+=n
 return dict(token_ce=total/count,reply_tokens=count,examples=len(examples))
def checkpoint(model,path,rewrite):
 model.save_pretrained(path);TOK.save_pretrained(path)
 (path/'v5_input.json').write_text(json.dumps(dict(speaker_labels=True,joint=False,rewrite_local=rewrite,max_source=256,max_target=64)))
def fit(name,train,valid,rewriter=False):
 torch.manual_seed(425);torch.cuda.manual_seed_all(425);random.seed(425)
 base=ROOT/'model';model=new_task_adapter(load_base(base))
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
 params=[p for p in model.parameters() if p.requires_grad]
 if any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):raise ValueError('Only task LoRA may train')
 if sum(p.numel() for p in params)!=1769472:raise ValueError('Expected rank16q/v adapter parameter count')
 mode='rewrite' if rewriter else 'direct';encoded=[encode(r,TOK,mode,256,64) for r in train];ve=[encode(r,TOK,mode,256,64) for r in valid]
 phase=dict(status='running',history=[],train_examples=len(train),validation_examples=len(valid),trainable_parameters=sum(p.numel() for p in params),selection='Minimum validation replyCE including epoch0',base_checkpoint=str(base),adapter_initialization='Fresh independent rank16q/v LoRA on the same frozen adapted base');R['phases'][name]=phase;save()
 opt=torch.optim.AdamW(params,lr=3e-5,weight_decay=.01,foreach=False);best=float('inf');stale=0;steps=0
 for epoch in range(13):
  if epoch:
   model.train();opt.zero_grad(set_to_none=True);order=torch.randperm(len(encoded),generator=torch.Generator().manual_seed(425+epoch)).tolist()
   # English effective batch4 (micro2 x accumulation2), renderer effective8.
   micro=1 if rewriter else 2;accum=8 if rewriter else 2;chunks=[order[i:i+micro] for i in range(0,len(order),micro)]
   for i,indices in enumerate(chunks):
    x,a,y=tensor([encoded[j] for j in indices]);loss=reply_loss(model(input_ids=x,attention_mask=a,labels=y,use_cache=False).logits,y)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    group=min(accum,len(chunks)-(i//accum)*accum);(loss/group).backward()
    if (i+1)%accum==0 or i+1==len(chunks):
     norm=torch.nn.utils.clip_grad_norm_(params,1.)
     if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
     opt.step();opt.zero_grad(set_to_none=True);steps+=1
     if steps==1:phase['first_update_finite']=True;save();print(name,'first optimizer update finite',flush=True)
  scores=validation(model,ve);phase['history'].append(dict(epoch=epoch,optimizer_steps=steps,validation=scores))
  if scores['token_ce']<best:
   best=scores['token_ce'];stale=0;phase.update(best_epoch=epoch,best_validation=scores);checkpoint(model,OUT/name/'best',rewriter)
  elif epoch:stale+=1
  phase['stale_epochs']=stale;save();print(name,epoch,scores,flush=True)
  if stale>=2:break
 phase['status']='complete';save();del opt,params,model;clear();return OUT/name/'best'
@torch.no_grad()
def generate(model,cases,style,method,anchors=None):
 model.eval();result=[]
 for start in range(0,len(cases),4):
  part=cases[start:start+4];ready=[];texts=[];answers={}
  for j,row in enumerate(part):
   anchor=None if anchors is None else anchors[start+j]
   if anchors is not None and not anchor.strip():answers[j]=dict(item_id=row['item_id'],text='',status='failed_empty_anchor',token_ids=[])
   else:ready.append(j);texts.append(source(row,style,anchor))
  if ready:
   inputs=TOK(texts,return_tensors='pt',padding=True)
   if inputs['input_ids'].shape[-1]>256:raise ValueError('No generation truncation')
   seq=model.generate(**{k:v.cuda() for k,v in inputs.items()},max_new_tokens=64,use_cache=True,**DECODE[method])
   for j,tokens in zip(ready,seq,strict=True):
    text=TOK.decode(tokens,skip_special_tokens=True);answers[j]=dict(item_id=part[j]['item_id'],text=text,token_ids=tokens.tolist(),status='failed_empty' if not text.strip() else 'complete' if TOK.eos_token_id in tokens.tolist() else 'length_limit')
  result.extend(answers[j] for j in range(len(part)))
 return result

def main():
 global TOK,R
 if (OUT/'results.json').exists():raise FileExistsError('Fresh snapshot required')
 recipe=read('recipe.json');manifest=read('source_manifest.json')
 for p,h in recipe['files_sha256'].items():
  if sha(ROOT/p)!=h:raise ValueError('Frozen source mismatch '+p)
 for p,h in manifest['files_sha256'].items():
  if sha(ROOT/'model'/p)!=h:raise ValueError('Adapted source checkpoint changed '+p)
 if not (ROOT/'data/training_expansion/generated600_repair_v1/combined_train.jsonl').read_bytes().startswith((ROOT/'data/dev/train.jsonl').read_bytes()):raise ValueError('Protected team data changed')
 torch.set_num_threads(4);TOK=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
 if 'L40S' not in torch.cuda.get_device_name(0):raise ValueError('L40S required')
 R=dict(status='running',recipe=recipe,source_manifest=manifest,phases={},outputs={},default_changed=False,environment=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,gpu=torch.cuda.get_device_name(0)));save()
 try:
  tr=rows('data/training_expansion/generated600_repair_v1/combined_train.jsonl');va=rows('data/dev/validation.jsonl');audit=read('data/training_expansion/sealion_rewrite_v1/pair_audit.json')
  standard=[r for r in task_rows(tr) if r['style']=='STANDARD'];vstandard=[r for r in task_rows(va) if r['style']=='STANDARD']
  if sha(ROOT/'data/training_expansion/generated600_repair_v1/combined_train.jsonl')!=audit['source_sha256']:raise ValueError('Pair audit source mismatch')
  if len(standard)!=790 or len(vstandard)!=50:raise ValueError('English pair counts')
  baseline=read('baseline_results.json')
  english=fit('standard',standard,vstandard)
  english_hashes={p.name:sha(p) for p in english.iterdir() if p.is_file()}
  reused=read('data/evaluation/sealion_full_rewrite_v1/inputs.json');fresh=read('data/evaluation/restored_pipeline_v1/inputs.json');allcases=reused+fresh
  model=activate_inference(PeftModel.from_pretrained(load_base(ROOT/'model'),english,adapter_name='english',is_trainable=False),'english')
  for method in DECODE:R['outputs'][method]=dict(english=generate(model,allcases,'STANDARD',method));save()
  del model;clear()
  train_ids={r['id'] for r in audit['items'] if r['split']=='train' and r['eligible']};val_ids={r['id'] for r in audit['items'] if r['split']=='validation' and r['eligible']}
  rt=paired_rows(tr,train_ids);rv=paired_rows(va,val_ids)
  if len(rt)!=745 or len(rv)!=42:raise ValueError('Audited rewrite counts')
  rewrite=fit('rewrite',rt,rv,True)
  if any(sha(english/p)!=h for p,h in english_hashes.items()):raise ValueError('English adapter changed during Singlish training')
  model=load_switching_adapters(load_base(ROOT/'model'),english,rewrite)
  R['adapter_integrity']=dict(english_files_before_singlish=english_hashes,english_unchanged_after_singlish_training=True,same_frozen_base=True,roundtrip={})
  challenge=read('data/training_expansion/sealion_fidelity_v1/challenge.json')
  for method in DECODE:
   activate_inference(model,'english')
   replay=generate(model,allcases,'STANDARD',method)
   if replay!=R['outputs'][method]['english']:raise ValueError('English outputs changed after loading/switching adapters')
   R['adapter_integrity']['roundtrip'][method]=dict(exact_english_replay=len(replay))
   activate_inference(model,'singlish')
   own=[x['text'] for x in R['outputs'][method]['english']]
   baseline_english=baseline['outputs'][method]['english']
   if [x['item_id'] for x in baseline_english]!=[x['item_id'] for x in allcases]:raise ValueError('Baseline evaluation IDs differ')
   groups=[('actual_english',allcases,own),('shared_mt5_english',allcases,[x['text'] for x in baseline_english]),('shared_sea_english',reused,[r['actual_anchor'] for r in reused]),('diagnostic',[r for r in reused if 'diagnostic_anchor' in r],[r['diagnostic_anchor'] for r in reused if 'diagnostic_anchor' in r]),('fresh_diagnostic',fresh,[r['diagnostic_anchor'] for r in fresh]),('challenge',challenge,[r['standard_anchor'] for r in challenge])]
   for name,cases,anchors in groups:
    outputs=generate(model,cases,'LOCAL_REWRITE',method,anchors);R['outputs'][method][name]=[dict(**o,standard_anchor=a) for o,a in zip(outputs,anchors,strict=True)];save()
  if any(sha(ROOT/'model'/p)!=h for p,h in manifest['files_sha256'].items()):raise ValueError('Original adapted source changed')
  flags=[dict(method=m,route=route,item_id=x['item_id'],status=x['status']) for m,routes in R['outputs'].items() for route,rs in routes.items() for x in rs if x['status']!='complete']
  R['comparison']=dict(baseline_job=356868,selection_and_evaluation='Same development rows, tokenizer, maximum epochs, early stopping, LR3e-5, batch sizes and decoders as restored full-English baseline. Adapter LR not separately tuned.',limitations='Copy counts describe style differentiation, not contextual correctness; evaluation is reused development, not independent human gold.',phases={name:dict(baseline_best_ce=baseline['phases'][name]['best_validation']['token_ce'],switch_best_ce=R['phases'][name]['best_validation']['token_ce']) for name in ['standard','rewrite']},copies={})
  for method in DECODE:
   R['comparison']['copies'][method]={}
   for subset,lo,hi in [('reused82',0,len(reused)),('fresh24',len(reused),len(allcases))]:
    R['comparison']['copies'][method][subset]={label:sum(e['text']==l['text'] for e,l in zip(result['outputs'][method]['english'][lo:hi],result['outputs'][method]['actual_english'][lo:hi],strict=True)) for label,result in [('baseline',baseline),('switching',R)]}
  if any(sha(english/p)!=h for p,h in english_hashes.items()):raise ValueError('English checkpoint changed during evaluation')
  R.update(status='complete_with_flags' if flags else 'complete',flagged_outputs=flags,original_checkpoint_unchanged=True);save()
 except Exception as e:R.update(status='failed',error=str(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
