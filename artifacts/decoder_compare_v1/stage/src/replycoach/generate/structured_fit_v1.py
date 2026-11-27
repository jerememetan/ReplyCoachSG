"""Three predeclared supervised fit probes for the restricted action task."""
import gc,json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from replycoach.generate.structured_rl_v1 import encode_cases,action_distribution,LETTERS,reward
from replycoach.generate.structured_rl_train_v1 import load_model,verifier_rewards
from replycoach.generate.finetune_v5 import sequence_logprobs
from replycoach.generate.train_v5 import collate,epoch_batches,validation_loss

MODES=('original_objective','plain_choice','seq2seq')


def scores(model,encoded,pad_id,device):
    return sequence_logprobs(model,[pair for options in encoded for pair in options],pad_id,device).reshape(len(encoded),4)


def gold_rows(cases,encoded):
    return [options[next(i for i,a in enumerate(c['actions']) if reward(c,a)==1)] for c,options in zip(cases,encoded)]


@torch.no_grad()
def evaluate(model,cases,encoded,pad_id,device,tokenizer,batch_size=2):
    model.eval();outputs=[]
    gold=gold_rows(cases,encoded)
    for start in range(0,len(cases),batch_size):
        batch=cases[start:start+batch_size];options=encoded[start:start+batch_size]
        logits=scores(model,options,pad_id,device);p=logits.softmax(-1)
        selected=p.argmax(-1).cpu().tolist()
        ids,mask,_=collate([o[0] for o in options],pad_id)
        generated=model.generate(input_ids=ids.to(device),attention_mask=mask.to(device),max_new_tokens=4,num_beams=1,do_sample=False)
        texts=tokenizer.batch_decode(generated,skip_special_tokens=True)
        for c,choice,text,probs in zip(batch,selected,texts,p.cpu().tolist()):
            correct=next(i for i,a in enumerate(c['actions']) if reward(c,a)==1)
            outputs.append(dict(id=c['id'],category=c['category'],chosen_letter=LETTERS[choice],correct_letter=LETTERS[correct],choice_correct=choice==correct,generated=text,direct_correct=text.strip()==LETTERS[correct],direct_valid=text.strip() in LETTERS,probabilities=probs))
    return dict(cases=len(cases),choice_correct=sum(r['choice_correct'] for r in outputs),direct_correct=sum(r['direct_correct'] for r in outputs),direct_valid=sum(r['direct_valid'] for r in outputs),gold_token_loss=validation_loss(model,gold,batch_size,pad_id,device),outputs=outputs)


def fit(base,train,permuted,novel,output,mode,device='cuda',steps=240,batch_size=2,lr=1e-4,seed=425,record_steps=(0,40,120,240)):
    if mode not in MODES:raise ValueError('Unknown diagnostic mode')
    if {c['id'] for c in train}&{c['id'] for c in novel}:raise ValueError('Novel case leakage')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    parts={'train':train,'permuted':permuted,'novel':novel}
    enc={name:encode_cases(rows,tokenizer) for name,rows in parts.items()}
    model=load_model(base,device)
    reference=None
    if mode=='original_objective':
        model.eval()
        with torch.no_grad():reference=torch.cat([action_distribution(scores(model,enc['train'][n:n+batch_size],tokenizer.pad_token_id,device)).cpu() for n in range(0,len(train),batch_size)])
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.01,foreach=False)
    report=dict(mode=mode,status='running',steps_requested=steps,learning_rate=lr,batch_size=batch_size,seed=seed,train_cases=len(train),history=[],limitation='Training-fit diagnostic intentionally tests memorization capacity; reordered cases share facts, novel cases reuse synthetic validation. No production quality or independent test claim.')
    def measure(step):
        entry=dict(optimizer_steps=step)
        for name,rows in parts.items():entry[name]=evaluate(model,rows,enc[name],tokenizer.pad_token_id,device,tokenizer,batch_size)
        report['history'].append(entry);report['optimizer_steps']=step
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(mode=mode,step=step,metrics={name:{k:v for k,v in entry[name].items() if k!='outputs'} for name in parts})),flush=True)
    measure(0);step=0;epoch=0
    while step<steps:
        epoch+=1
        for picks in epoch_batches(len(train),batch_size,seed,epoch):
            model.train();optimizer.zero_grad(set_to_none=True)
            cases=[train[i] for i in picks];batch=[enc['train'][i] for i in picks]
            if mode=='seq2seq':
                ids,mask,labels=collate(gold_rows(cases,batch),tokenizer.pad_token_id)
                loss=model(input_ids=ids.to(device),attention_mask=mask.to(device),labels=labels.to(device)).loss
            else:
                logits=scores(model,batch,tokenizer.pad_token_id,device)
                targets=verifier_rewards(cases,device).argmax(-1,keepdim=True)
                if mode=='plain_choice':loss=-logits.log_softmax(-1).gather(1,targets).mean()
                else:
                    p=action_distribution(logits);ref=reference[picks].to(device)
                    kl=(p*(p.log()-ref.log())).sum(-1).mean()
                    loss=-p.log().gather(1,targets).mean()+.05*kl
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            optimizer.step();step+=1
            if step in record_steps or step==steps:measure(step)
            if step>=steps:break
    model.save_pretrained(output/'final');tokenizer.save_pretrained(output/'final')
    (output/'final/structured_policy.json').write_text(json.dumps(dict(task='Tiny structured-fit diagnostic',mode=mode)))
    report['status']='complete';(output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    del optimizer,parameters,model;gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return report
