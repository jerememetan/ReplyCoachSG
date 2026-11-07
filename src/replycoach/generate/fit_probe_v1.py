"""Small STANDARD-only fit diagnostic, separate from production paired-style training."""
from __future__ import annotations
import gc,json,math,re,time
from pathlib import Path
import torch
from peft import LoraConfig,get_peft_model
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.train_v5 import collate,encode_rows,epoch_batches,generate_replies,validation_loss
from replycoach.generate.style_control import mt5_input

def normalize_reply(text):
    return re.sub(r'\s+',' ',text.strip().casefold()).rstrip('.!?')

def checked_encoding(rows,tokenizer,max_source=192,max_target=48):
    if not rows or any(r.get('style')!='STANDARD' for r in rows):
        raise ValueError('Fit probe requires nonempty STANDARD-only rows')
    if len({r['item_id'] for r in rows})!=len(rows):raise ValueError('Duplicate fit-probe IDs')
    for r in rows:
        if not r.get('response','').strip():raise ValueError('Nonempty target required')
        text=mt5_input('STANDARD',r['message'],r['context'],True,r['speakers'])
        if len(tokenizer(text)['input_ids'])>max_source or len(tokenizer(r['response'])['input_ids'])>max_target:
            raise ValueError('Fit-probe input/target exceeds untruncated token budget')
    return encode_rows(rows,tokenizer,max_source,max_target,speaker_labels=True)

def generate_standard(model,tokenizer,rows,device,batch_size=4,max_source=192,max_target=48):
    replies=[]
    for start in range(0,len(rows),batch_size):
        replies.extend(generate_replies(model,tokenizer,rows[start:start+batch_size],'STANDARD',max_source,max_target,device,speaker_labels=True))
    return [dict(item_id=r['item_id'],generated_standard=text,target=r['response'],exact_target_match=normalize_reply(text)==normalize_reply(r['response'])) for r,text in zip(rows,replies)]

def fit(base_model,rows,output_dir,method,learning_rate=1e-4,steps=400,batch_size=4,seed=425,device=None,record_steps=(0,100,400)):
    if method not in ('lora','full'):raise ValueError('method must be lora or full')
    if type(steps) is not int or steps<1 or batch_size<1 or learning_rate<=0:raise ValueError('Invalid fit budget/settings')
    output_dir=Path(output_dir)
    if output_dir.exists():raise FileExistsError('Fresh diagnostic run directory required')
    torch.manual_seed(seed)
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    tokenizer=AutoTokenizer.from_pretrained(base_model,local_files_only=True)
    encoded=checked_encoding(rows,tokenizer)
    model=AutoModelForSeq2SeqLM.from_pretrained(base_model,local_files_only=True).to(device)
    if method=='lora':
        model=get_peft_model(model,LoraConfig(task_type='SEQ_2_SEQ_LM',r=16,lora_alpha=32,lora_dropout=.1,target_modules=['q','v'])).to(device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    count=sum(p.numel() for p in parameters);total=sum(p.numel() for p in model.parameters())
    if method=='full' and count!=total:raise ValueError('Full diagnostic must train every parameter')
    optimizer=torch.optim.AdamW(parameters,lr=learning_rate,weight_decay=.01,foreach=False)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    output_dir.mkdir(parents=True)
    report=dict(method=method,learning_rate=learning_rate,seed=seed,steps_requested=steps,batch_size=batch_size,train_items=len(rows),training_style='STANDARD',trainable_parameters=count,total_parameters=total,device=str(device),device_name=torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',optimizer='AdamW, weight_decay0.01, foreach=False',history=[],train_generations={})
    started=time.perf_counter()
    def measure(step):
        loss=validation_loss(model,encoded,batch_size,tokenizer.pad_token_id,device)
        outputs=generate_standard(model,tokenizer,rows,device,batch_size)
        entry=dict(optimizer_steps=step,teacher_forced_train_loss=loss,training_exact_matches=sum(r['exact_target_match'] for r in outputs),seconds=time.perf_counter()-started)
        report['history'].append(entry);report['train_generations'][str(step)]=outputs
        report['optimizer_steps']=step
        if device.type=='cuda':report['peak_cuda_gb']=torch.cuda.max_memory_allocated()/1024**3
        (output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(entry),flush=True)
    measure(0)
    step=0;epoch=0
    while step<steps:
        epoch+=1
        for indices in epoch_batches(len(encoded),batch_size,seed,epoch):
            model.train();optimizer.zero_grad(set_to_none=True)
            ids,mask,labels=collate([encoded[i] for i in indices],tokenizer.pad_token_id)
            loss=model(input_ids=ids.to(device),attention_mask=mask.to(device),labels=labels.to(device)).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; preserve failed run without hiding it')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.0)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient norm')
            optimizer.step();step+=1
            if step in record_steps or step==steps:measure(step)
            if step>=steps:break
    report['epochs_started']=epoch
    checkpoint=output_dir/'final';model.save_pretrained(checkpoint);tokenizer.save_pretrained(checkpoint)
    (checkpoint/'v5_input.json').write_text(json.dumps(dict(speaker_labels=True,joint=False,standard_only_diagnostic=True,max_source=192,max_target=48))+'\n')
    report['checkpoint']=str(checkpoint);report['status']='training_complete'
    (output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    del optimizer,parameters
    model.zero_grad(set_to_none=True);gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return report,model,tokenizer
