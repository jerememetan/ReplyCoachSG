"""Finite-action REINFORCE with exact categorical reference KL; matched SFT control."""
import gc,json,math
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from replycoach.generate.structured_rl_v1 import encode_cases,reward,action_distribution,LETTERS
from replycoach.generate.finetune_v5 import sequence_logprobs


def policy(model,batch,pad_id,device):
    scores=sequence_logprobs(model,[pair for options in batch for pair in options],pad_id,device).reshape(len(batch),4)
    return action_distribution(scores)


def verifier_rewards(cases,device):
    values=torch.tensor([[reward(c,a) for a in c['actions']] for c in cases],dtype=torch.float32,device=device)
    if not ((values==1).sum(-1)==1).all():raise ValueError('Exactly one feasible action required')
    return values


def reinforce_objective(probabilities,reference,rewards,group_size=8,kl_weight=.05):
    if group_size<2:raise ValueError('Leave-one-out baseline requires at least2 samples')
    picks=torch.multinomial(probabilities.detach(),group_size,replacement=True)
    sampled=rewards.gather(1,picks)
    baseline=(sampled.sum(-1,keepdim=True)-sampled)/(group_size-1)
    advantages=(sampled-baseline).detach()
    logp=probabilities.log().gather(1,picks)
    kl=(probabilities*(probabilities.log()-reference.log())).sum(-1).mean()
    return -(advantages*logp).mean()+kl_weight*kl,dict(kl=float(kl.detach()),sampled_mean_reward=float(sampled.mean()))


@torch.no_grad()
def evaluate(model,cases,encoded,pad_id,device,batch_size=2):
    model.eval();outputs=[];correct=expected=0.
    for start in range(0,len(cases),batch_size):
        batch=cases[start:start+batch_size];p=policy(model,encoded[start:start+batch_size],pad_id,device)
        rewards=verifier_rewards(batch,device);selected=p.argmax(-1)
        good=rewards.argmax(-1)
        correct+=float((selected==good).sum());expected+=float((p*rewards).sum())
        for c,probs,choice in zip(batch,p.cpu().tolist(),selected.cpu().tolist()):
            outputs.append(dict(id=c['id'],category=c['category'],selected_letter=LETTERS[choice],selected_action=c['actions'][choice],verified_reward=reward(c,c['actions'][choice]),probabilities=probs))
    return dict(cases=len(cases),greedy_correct=int(correct),greedy_accuracy=correct/len(cases),expected_reward=expected/len(cases),outputs=outputs)


def load_model(base,device):
    model=AutoModelForSeq2SeqLM.from_pretrained(base,local_files_only=True).to(device)
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    return model


def fit(base,train,validation,test,output,arm,device='cuda',epochs=3,batch_size=2,lr=3e-5,seed=425):
    if arm not in ('rl','sft'):raise ValueError('Unknown arm')
    groups=[{json.dumps(c['facts'],sort_keys=True) for c in split} for split in (train,validation,test)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('Shared fact state across partitions')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    encoded=encode_cases(train,tokenizer);valid=encode_cases(validation,tokenizer)
    model=load_model(base,device);model.eval()
    with torch.no_grad():
        reference=torch.cat([policy(model,encoded[i:i+batch_size],tokenizer.pad_token_id,device).cpu() for i in range(0,len(encoded),batch_size)])
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.01,foreach=False)
    report=dict(arm=arm,status='running',task='restricted structured action selection; not conversational quality',optimizer='AdamW weight_decay.01 full parameters',learning_rate=lr,seed=seed,max_epochs=epochs,batch_size=batch_size,group_size=8,reference_kl_weight=.05,policy='Normalized full sequence option likelihoods, temperature2, .1 uniform exploration; this same distribution is sampled and differentiated',history=[],best_epoch=0,optimizer_steps=0)
    best=-float('inf');stale=0;steps=0
    for epoch in range(epochs+1):
        diagnostics=[]
        if epoch:
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
            for start in range(0,len(order),batch_size):
                picks=order[start:start+batch_size];batch=[train[i] for i in picks]
                model.train();optimizer.zero_grad(set_to_none=True)
                p=policy(model,[encoded[i] for i in picks],tokenizer.pad_token_id,device)
                ref=reference[picks].to(device);rewards=verifier_rewards(batch,device)
                if arm=='rl':loss,diag=reinforce_objective(p,ref,rewards)
                else:
                    kl=(p*(p.log()-ref.log())).sum(-1).mean()
                    loss=-p.log().gather(1,rewards.argmax(-1,keepdim=True)).mean()+.05*kl
                    diag=dict(kl=float(kl.detach()))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                optimizer.step();steps+=1;diagnostics.append(diag)
        train_metrics=evaluate(model,train,encoded,tokenizer.pad_token_id,device,batch_size)
        val_metrics=evaluate(model,validation,valid,tokenizer.pad_token_id,device,batch_size)
        entry=dict(epoch=epoch,optimizer_steps=steps,train={k:v for k,v in train_metrics.items() if k!='outputs'},validation={k:v for k,v in val_metrics.items() if k!='outputs'})
        if diagnostics:entry['mean_kl']=sum(d['kl'] for d in diagnostics)/len(diagnostics)
        report['history'].append(entry)
        if val_metrics['expected_reward']>best+1e-6:
            best=val_metrics['expected_reward'];stale=0;report['best_epoch']=epoch
            model.save_pretrained(output/'best');tokenizer.save_pretrained(output/'best')
            (output/'best/structured_policy.json').write_text(json.dumps(dict(task=report['task'],options=LETTERS,temperature=2.,uniform_exploration=.1)))
        elif epoch:stale+=1
        report.update(best_validation_expected_reward=best,optimizer_steps=steps)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(arm=arm,**entry)),flush=True)
        if stale>=1:break
    report['stop_reason']='validation_patience1' if stale else 'max_epochs'
    del optimizer,parameters,model;gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    # Synthetic test evaluated once after this arm's checkpoint is selected.
    model=load_model(output/'best',device)
    report['test']=evaluate(model,test,encode_cases(test,tokenizer),tokenizer.pad_token_id,device,batch_size)
    report['status']='complete'
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    del model;gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return report
