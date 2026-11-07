"""Fixed-budget small-set fitting diagnostic; never select on probe outputs."""
import hashlib,json,os,random
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.native_rewrite_v2 import encode,generate,normalized
from replycoach.generate.train_v5 import collate,validation_loss

def validate(train,val):
 if len(train)!=24 or len(val)!=12:raise ValueError('Frozen diagnostic sizes changed')
 if {r['group_id'] for r in train}&{r['group_id'] for r in val}:raise ValueError('Author overlap')
 if {normalized(r['local']) for r in train}&{normalized(r['local']) for r in val}:raise ValueError('Target overlap')
 if any(r['split']!='train' for r in train) or any(r['split']!='validation' for r in val):raise ValueError('Wrong partitions')

def main():
 root=Path('.');recipe=json.loads((root/'recipe.json').read_text())
 for f,digest in recipe['files_sha256'].items():
  if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=digest:raise ValueError(f'Snapshot hash mismatch: {f}')
 folder=root/'data/training_expansion/rewrite_diagnostic_v3';load=lambda name:[json.loads(s) for s in (folder/f'{name}.jsonl').read_text().splitlines()]
 train,val=load('train'),load('validation');validate(train,val);probes=json.loads((folder/'meaning_probes.json').read_text())
 base=Path(os.environ['PROJECT_ROOT'])/'artifacts/mt5_base_after_spans_spk/best';h=hashlib.sha256()
 with (base/'model.safetensors').open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 if h.hexdigest()!=recipe['base_sha256']:raise ValueError('Base mismatch')
 torch.manual_seed(425);random.seed(425)
 tok=AutoTokenizer.from_pretrained(base,local_files_only=True);model=AutoModelForSeq2SeqLM.from_pretrained(base,local_files_only=True).to('cuda')
 encoded=encode(train,tok,'anchor_only',None);valid=encode(val,tok,'anchor_only',None)
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,foreach=False)
 out=Path('artifacts');out.mkdir(exist_ok=False)
 report=dict(status='running',purpose='Intentional small-set capacity diagnostic, not product model selection',settings=dict(steps=300,batch_size=4,lr=1e-4,seed=425,mode='anchor_only',base=str(base)),history=[],limitations=json.loads((folder/'manifest.json').read_text())['limitations'])
 def record(step):
  model.eval();entry=dict(step=step,train_loss=validation_loss(model,encoded,4,tok.pad_token_id,'cuda'),validation_loss=validation_loss(model,valid,4,tok.pad_token_id,'cuda'))
  if step in (0,100,300):
   for name,rows in [('train',train),('validation',val),('meaning_probes',probes)]:
    outputs=generate(model,tok,rows,'anchor_only',None,'cuda',4)
    entry[name+'_outputs']=outputs
    if name!='meaning_probes':entry[name+'_normalized_exact_match']=sum(normalized(x['local'])==normalized(r['local']) for x,r in zip(outputs,rows))/len(rows)
  report['history'].append(entry);(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps({k:v for k,v in entry.items() if not k.endswith('_outputs')}),flush=True)
 record(0)
 order=[]
 for step in range(1,301):
  if not order:order=torch.randperm(len(encoded)).tolist()
  picks=order[:4];order=order[4:];ids,mask,labels=collate([encoded[i] for i in picks],tok.pad_token_id)
  model.train();optimizer.zero_grad(set_to_none=True);loss=model(input_ids=ids.cuda(),attention_mask=mask.cuda(),labels=labels.cuda()).loss
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
  optimizer.step()
  if step%25==0:record(step)
 model.save_pretrained(out/'final');tok.save_pretrained(out/'final')
 report.update(status='complete',checkpoint_policy='Fixed step 300 capacity endpoint; not a validation-selected production checkpoint')
 (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
