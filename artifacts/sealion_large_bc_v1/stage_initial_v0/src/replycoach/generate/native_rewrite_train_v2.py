"""Matched native-target LOCAL ablations with grouped validation checkpointing."""
import gc,json,math,random
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.native_rewrite_v2 import MODES,StyleExamples,encode,generate
from replycoach.generate.train_v5 import collate,validation_loss


def fit(base,train,val,anchors,out,mode,device='cuda',lr=3e-5,batch_size=4,max_epochs=8,patience=3,seed=425):
    if mode not in MODES:raise ValueError('Unknown mode')
    if not train or not val or any(r['split']!='train' for r in train) or any(r['split']!='validation' for r in val):raise ValueError('Explicit train and validation partitions required')
    if {r['group_id'] for r in train}&{r['group_id'] for r in val}:raise ValueError('Group leakage')
    if {r['item_id'] for r in train}&{r['item_id'] for r in anchors}:raise ValueError('Generation audit leaked into train')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    retriever=StyleExamples(train) if mode=='exemplar' else None
    encoded=encode(train,tokenizer,mode,retriever)
    valid=encode(val,tokenizer,mode,retriever)
    # Preflight every declared anchor variant and generated evaluation source, without truncation.
    for row in train:
        encode([{**row,'standard':s} for s in row['standard_variants']],tokenizer,mode,retriever)
    from replycoach.generate.native_rewrite_v2 import source
    for row in anchors:
        if len(tokenizer(source(row,mode,retriever)[0])['input_ids'])>384:raise ValueError('Fresh generated anchor exceeds input budget')
    model=AutoModelForSeq2SeqLM.from_pretrained(base,local_files_only=True).to(device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.01,foreach=False)
    indices={'original':[i for i,r in enumerate(val) if r['item_id'].startswith('ITEM-')],'public':[i for i,r in enumerate(val) if r['item_id'].startswith('NATIVE-PUB-')]}
    report=dict(status='running',mode=mode,base=str(base),train_items=len(train),validation_items=len(val),max_epochs=max_epochs,learning_rate=lr,batch_size=batch_size,seed=seed,patience=patience,min_delta=.0001,history=[],optimizer_steps=0,selection='Minimum macro mean of original-author-heldout and CoSEM-chat-heldout target token losses, baseline included',limitation='Assistant-audited native-target normalizations, not human gold. Component validation is not pristine end-to-end student quality evidence. Fresh40 conversational generations have no reference targets.')
    best=float('inf');stale=0;step=0
    @torch.no_grad()
    def measure(epoch):
        nonlocal best,stale
        model.eval();losses={name:validation_loss(model,[valid[i] for i in ix],batch_size,tokenizer.pad_token_id,device) for name,ix in indices.items() if ix}
        average=sum(losses.values())/len(losses)
        if not math.isfinite(average):raise RuntimeError('Nonfinite validation loss')
        entry=dict(epoch=epoch,optimizer_steps=step,train_token_loss=validation_loss(model,encoded,batch_size,tokenizer.pad_token_id,device),validation=losses,validation_macro=average)
        report['history'].append(entry)
        if average<best-.0001:
            best=average;stale=0;report['best_epoch']=epoch;report['best_step']=step
            model.save_pretrained(out/'best');tokenizer.save_pretrained(out/'best')
            (out/'best/native_rewrite_config.json').write_text(json.dumps(dict(mode=mode,max_source=384,max_target=64,trained_task='Native-target LOCAL rewrite only; use native_rewrite_v2 serializer, not legacy provider input'))+'\n')
            if mode=='exemplar':(out/'best/style_pool.json').write_text(json.dumps(train,ensure_ascii=False)+'\n')
        elif step:stale+=1
        report.update(best_validation_macro=best,stale_evaluations=stale,optimizer_steps=step)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(mode=mode,**entry)),flush=True)
    measure(0)
    for epoch in range(1,max_epochs+1):
        rng=random.Random(seed+epoch)
        epoch_rows=[{**r,'standard':rng.choice(r['standard_variants'])} for r in train]
        epoch_encoded=encode(epoch_rows,tokenizer,mode,retriever)
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        batches=[order[i:i+batch_size] for i in range(0,len(order),batch_size)]
        checkpoints={max(1,len(batches)//2),len(batches)}
        for batch_number,picks in enumerate(batches,1):
            model.train();optimizer.zero_grad(set_to_none=True)
            ids,mask,labels=collate([epoch_encoded[i] for i in picks],tokenizer.pad_token_id)
            loss=model(input_ids=ids.to(device),attention_mask=mask.to(device),labels=labels.to(device)).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
            optimizer.step();step+=1
            if batch_number in checkpoints:
                measure(epoch-1+batch_number/len(batches))
                if stale>=patience:break
        if stale>=patience:break
    report['stop_reason']='early_stopping' if stale>=patience else 'max_epochs'
    del optimizer,parameters,model;gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    model=AutoModelForSeq2SeqLM.from_pretrained(out/'best',local_files_only=True).to(device).eval()
    # Generate reference-anchor validation diagnostics, then fixed generated-anchor fresh outputs.
    report['reference_anchor_validation_outputs']=generate(model,tokenizer,val,mode,retriever,device,batch_size)
    report['fresh_outputs']=generate(model,tokenizer,anchors,mode,retriever,device,batch_size)
    assert all(r['standard']==r['local_conditioned_on'] for r in report['fresh_outputs'])
    report['status']='complete';(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report
