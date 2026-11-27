"""Conservative full-parameter DPO pilot with grouped validation preferences."""
import gc,json,math,time
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.fit_probe_v1 import checked_encoding,generate_standard
from replycoach.generate.finetune_v5 import sequence_logprobs,dpo_loss
from replycoach.generate.train_v5 import collate,epoch_batches
from replycoach.generate.early_stopping import EarlyStopping


def encode_preferences(rows,tokenizer):
    def encode(row,field):
        return checked_encoding([dict(item_id=row['item_id'],style='STANDARD',context=row['context'],speakers=row['speakers'],message=row['message'],response=row[field])],tokenizer)[0]
    return [(encode(r,'chosen'),encode(r,'rejected')) for r in rows]


def fit(base,train,val,audit,output,device='cuda',max_epochs=5,batch_size=2,lr=1e-5,beta=.1,anchor_weight=.1,seed=425):
    if not train or not val:raise ValueError('Both preference partitions required')
    train_ids={r['item_id'] for r in train};val_ids={r['item_id'] for r in val}
    if train_ids&val_ids:raise ValueError('Conversation leakage between train and validation')
    if train_ids&{r['item_id'] for r in audit}:raise ValueError('Audit includes DPO training conversations')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    encoded=encode_preferences(train,tokenizer);valid=encode_preferences(val,tokenizer)
    model=AutoModelForSeq2SeqLM.from_pretrained(base,local_files_only=True).to(device)
    # Disable dropout identically for cached reference and trainable policy.
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.0
    model.eval()
    @torch.no_grad()
    def reference(pairs):
        chunks=[]
        for start in range(0,len(pairs),batch_size):
            batch=pairs[start:start+batch_size];n=len(batch)
            logs=sequence_logprobs(model,[p[0] for p in batch]+[p[1] for p in batch],tokenizer.pad_token_id,device)
            chunks.append(torch.stack([logs[:n],logs[n:]],dim=1).detach().cpu())
        return torch.cat(chunks)
    ref=reference(encoded);valref=reference(valid)
    report=dict(status='running',seed=seed,learning_rate=lr,beta=beta,chosen_sft_weight=anchor_weight,max_epochs=max_epochs,patience=2,train_pairs=len(train),validation_pairs=len(val),train_conversations=len(train_ids),validation_conversations=len(val_ids),reference='Frozen pre-update epoch1 checkpoint; cached full sequence log probabilities',history=[],baseline_generations=generate_standard(model,tokenizer,audit,device,4),limitation='Grouped held-out preferences from reused development cases, one seed, small pilot; not unbiased test performance or LOCAL assessment.')
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.01,foreach=False)
    selector=EarlyStopping(patience=2);steps=0
    @torch.no_grad()
    def measure(pairs,references):
        model.eval();loss_sum=agreement_sum=raw_sum=0
        for start in range(0,len(pairs),batch_size):
            batch=pairs[start:start+batch_size];n=len(batch)
            loss,agree=dpo_loss(model,batch,references[start:start+n],tokenizer.pad_token_id,device,beta)
            logs=sequence_logprobs(model,[p[0] for p in batch]+[p[1] for p in batch],tokenizer.pad_token_id,device)
            loss_sum+=float(loss)*n;agreement_sum+=agree*n;raw_sum+=int((logs[:n]>logs[n:]).sum())
        return dict(dpo_loss=loss_sum/len(pairs),relative_preference_agreement=agreement_sum/len(pairs),raw_chosen_more_likely=raw_sum/len(pairs))
    for epoch in range(max_epochs+1):
        if epoch:
            for picks in epoch_batches(len(encoded),batch_size,seed,epoch):
                model.train();optimizer.zero_grad(set_to_none=True)
                batch=[encoded[i] for i in picks]
                preference,_=dpo_loss(model,batch,ref[picks],tokenizer.pad_token_id,device,beta)
                ids,mask,labels=collate([p[0] for p in batch],tokenizer.pad_token_id)
                anchor=model(input_ids=ids.to(device),attention_mask=mask.to(device),labels=labels.to(device)).loss
                loss=preference+anchor_weight*anchor
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite DPO objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite DPO gradients')
                optimizer.step();steps+=1
        entry=dict(epoch=epoch,optimizer_steps=steps,train=measure(encoded,ref),validation=measure(valid,valref))
        report['history'].append(entry)
        if selector.observe(epoch,entry['validation']['dpo_loss']):
            model.save_pretrained(output/'best');tokenizer.save_pretrained(output/'best')
            (output/'best/v5_input.json').write_text(json.dumps(dict(speaker_labels=True,joint=False,standard_only_diagnostic=True,max_source=192,max_target=48)))
        report.update(best_epoch=selector.best_epoch,best_validation_dpo_loss=selector.best,optimizer_steps=steps)
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(entry),flush=True)
        if selector.should_stop:break
    report['stop_reason']='early_stopping' if selector.should_stop else 'max_epochs'
    del optimizer,parameters,model;gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    model=AutoModelForSeq2SeqLM.from_pretrained(output/'best',local_files_only=True).to(device)
    report['dpo_generations']=generate_standard(model,tokenizer,audit,device,4)
    report['status']='complete'
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report
