"""Audit old endpoints, then retrain with per-epoch validation and early stopping."""
import gc,hashlib,json,math,os,sys
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from peft import PeftModel,LoraConfig,get_peft_model
from replycoach.generate.early_stopping import EarlyStopping
from replycoach.generate.fit_probe_v1 import checked_encoding,generate_standard
from replycoach.generate.finetune_v5 import item_rows
from replycoach.generate.train_v5 import validation_loss,collate,epoch_batches

variant=sys.argv[1]
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
method,rate=recipe['variants'][variant]
base=Path(os.environ['FOCUS_BASE']);old=Path(os.environ['OLD_ROOT'])/'artifacts'/variant/'final'
out=Path('artifacts')/variant
out.mkdir(parents=True,exist_ok=False)
tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
train=[r for r in item_rows(Path(recipe['training_file'])) if r['style']=='STANDARD']
val=[r for r in item_rows(Path('data/dev/validation.jsonl')) if r['style']=='STANDARD']
assert len(train)==790 and len(val)==50
assert not {r['item_id'] for r in train}&{r['item_id'] for r in val}
encoded=checked_encoding(train,tokenizer);valid=checked_encoding(val,tokenizer)
def losses(model):
    with torch.no_grad():
        values=dict(train_loss=validation_loss(model,encoded,4,tokenizer.pad_token_id,'cuda'),validation_loss=validation_loss(model,valid,4,tokenizer.pad_token_id,'cuda'))
    assert all(math.isfinite(v) for v in values.values())
    return values
model=AutoModelForSeq2SeqLM.from_pretrained(base if method=='lora' else old,local_files_only=True).to('cuda')
if method=='lora':model=PeftModel.from_pretrained(model,old,is_trainable=False)
audit=losses(model)
(out/'old_endpoint_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(old_endpoint=variant,**audit)),flush=True)
del model;gc.collect();torch.cuda.empty_cache()
torch.manual_seed(425)
model=AutoModelForSeq2SeqLM.from_pretrained(base,local_files_only=True).to('cuda')
if method=='lora':model=get_peft_model(model,LoraConfig(task_type='SEQ_2_SEQ_LM',r=16,lora_alpha=32,lora_dropout=.1,target_modules=['q','v']))
params=[p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.AdamW(params,lr=rate,weight_decay=.01,foreach=False)
report=dict(variant=variant,old_endpoint=audit,history=[],best_epoch=0,patience=2,max_epochs=12,min_delta=0.0,status='running',selection='Minimum token-averaged development cross entropy, including epoch0 baseline',limitation='Repeatedly reused 50-item development set; not an unbiased test or semantic quality score. No LOCAL assessment or automatic promotion.')
selector=EarlyStopping(patience=2);steps=0
for epoch in range(13):
    if epoch:
        for indices in epoch_batches(len(encoded),4,425,epoch):
            model.train();optimizer.zero_grad(set_to_none=True)
            ids,mask,labels=collate([encoded[i] for i in indices],tokenizer.pad_token_id)
            loss=model(input_ids=ids.cuda(),attention_mask=mask.cuda(),labels=labels.cuda()).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.0)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
            optimizer.step();steps+=1
    values=losses(model);entry=dict(epoch=epoch,optimizer_steps=steps,**values)
    report['history'].append(entry)
    if selector.observe(epoch,values['validation_loss']):
        report['best_epoch']=epoch
        model.save_pretrained(out/'best');tokenizer.save_pretrained(out/'best')
        (out/'best/v5_input.json').write_text(json.dumps(dict(speaker_labels=True,joint=False,standard_only_diagnostic=True,max_source=192,max_target=48)))
    report['best_validation_loss']=selector.best;report['stale_epochs']=selector.stale
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(entry),flush=True)
    if selector.should_stop:break
report['stop_reason']='early_stopping' if selector.should_stop else 'max_epochs'
del optimizer,params,model;gc.collect();torch.cuda.empty_cache()
model=AutoModelForSeq2SeqLM.from_pretrained(base if method=='lora' else out/'best',local_files_only=True).to('cuda')
if method=='lora':model=PeftModel.from_pretrained(model,out/'best',is_trainable=False)
report['development_generations']=generate_standard(model,tokenizer,val,'cuda',4)
report['status']='complete'
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(variant=variant,status='complete',best_epoch=report['best_epoch'],best_validation_loss=selector.best)),flush=True)
