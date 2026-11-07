"""Train a small matched-control pilot of the user's mT5 reply-act architecture."""
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from replycoach.generate.reply_act_v1 import ReplyActMT5, REPLY_ACTS, act_vector, predicted_acts
from replycoach.generate.train_v5 import collate, epoch_batches
from replycoach.generate.style_control import mt5_input
from replycoach.generate.v5_provider import V5Generator, DECODING
from replycoach.generate.rewrite_v5 import rewrite_standard

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts'
BASE = Path(os.environ.get('ACT_BASE', '/common/home/users/h/hm.teoh.2024/ReplyCoachSG_standard_val_v1/artifacts/full_lr3e4/best'))
REWRITER = Path('/common/home/users/h/hm.teoh.2024/ReplyCoachSG_hybrid_inference_v1/checkpoints/local_rewrite')
ARMS = {'direct_control': (False, False), 'auxiliary_only': (False, True), 'conditioned': (True, True)}


def read(path):
    return json.loads((ROOT/path).read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)


def rows(split):
    labels=read(f'data/training_preferences/mt5_reply_act_v1/{split}_acts.json')
    items=[json.loads(l) for l in (ROOT/f'data/dev/{split}.jsonl').read_text().splitlines() if l.strip()]
    by_id={r['item_id']:r for r in labels}
    if len(by_id)!=len(labels) or set(by_id)!={r['id'] for r in items}:raise ValueError('Annotation coverage mismatch')
    result=[]
    for item in items:
        label=by_id[item['id']]
        if item['split']!='dev' or label['target_standard']!=item['replies']['standard']:raise ValueError('Wrong split or stale annotation')
        result.append(dict(item_id=item['id'],context=[t['text'] for t in item['session_memory']],speakers=[t['speaker'] for t in item['session_memory']],message=item['message']['text'],response=item['replies']['standard'],reply_acts=label['reply_acts']))
    return result


def source(row):
    return mt5_input('STANDARD',row['message'],row['context'],True,row['speakers'])


def encode(data,tok,source_limit=192,target_limit=48):
    result=[]
    for r in data:
        src=tok(source(r))['input_ids']; tgt=tok(r['response'])['input_ids']
        if len(src)>source_limit or len(tgt)>target_limit:raise ValueError(f"No silent truncation: {r['item_id']} lengths {len(src)}/{len(tgt)}")
        result.append((src,tgt,act_vector(r['reply_acts'])))
    return result


def batch(encoded,indices,tok):
    data=[encoded[i] for i in indices]
    ids,mask,labels=collate([(s,t) for s,t,_ in data],tok.pad_token_id)
    return ids.cuda(),mask.cuda(),labels.cuda(),torch.tensor([a for _,_,a in data],device='cuda')


@torch.no_grad()
def evaluate(model,data,tok,condition,auxiliary):
    model.eval(); total=0.; tokens=0; targets=[]; predictions=[]
    for start in range(0,len(data),2):
        ids,mask,labels,acts=batch(data,list(range(start,min(start+2,len(data)))),tok)
        out=model(ids,mask,labels,acts,condition,auxiliary)
        n=int(labels.ne(-100).sum());total+=float(out['token_loss'])*n;tokens+=n
        targets.append(acts.bool().cpu());predictions.append(out['act_logits'].sigmoid().ge(.5).cpu())
    truth=torch.cat(targets);pred=torch.cat(predictions)
    tp=(pred & truth).sum(0);fp=(pred & ~truth).sum(0);fn=(~pred & truth).sum(0)
    denom=2*tp+fp+fn
    return dict(token_ce=total/tokens,act_exact_match=float((pred==truth).all(-1).float().mean()),act_micro_f1=float(2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()).clamp_min(1)),act_macro_f1=float((2*tp/denom.clamp_min(1)).mean()),per_act={a:dict(trainable_class_index=i,validation_support=int(truth[:,i].sum()),f1=float(2*tp[i]/denom[i].clamp_min(1))) for i,a in enumerate(REPLY_ACTS)},classification_uses_raw_threshold=True)


@torch.no_grad()
def generate(model,tok,data,route):
    model.eval(); result=[]
    for r in data:
        ids=tok(source(r),return_tensors='pt')['input_ids'].cuda()
        if ids.shape[-1]>192:raise ValueError('Generation source budget exceeded')
        mask=torch.ones_like(ids)
        oracle=torch.tensor([act_vector(r['reply_acts'])],device='cuda') if route=='oracle' else None
        output,probs=model.generate(ids,mask,route,oracle,max_new_tokens=48,**DECODING)
        result.append(dict(item_id=r['item_id'],standard=tok.decode(output[0],skip_special_tokens=True),predicted_acts=predicted_acts(probs)[0],act_probabilities=dict(zip(REPLY_ACTS,probs[0].cpu().tolist())),conditioning=route,**({'oracle_acts':r['reply_acts']} if route=='oracle' else {})))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise FileExistsError('Fresh experiment output required')
    recipe=read('recipe.json')
    for path,digest in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source mismatch: '+path)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    train,val=rows('train'),rows('validation')
    if {r['item_id'] for r in train}&{r['item_id'] for r in val}:raise ValueError('Train/validation overlap')
    diagnostics=read('data/evaluation/component_baseline_v1/inputs.json')
    oracle=read('data/training_preferences/mt5_reply_act_v1/diagnostic_acts.json')
    by_id={r['item_id']:r['reply_acts'] for r in oracle}
    diagnostics=[{**r,'response':'','reply_acts':by_id[r['item_id']]} for r in diagnostics]
    tok=AutoTokenizer.from_pretrained(BASE)
    tr=encode(train,tok);va=encode(val,tok)
    result=dict(status='running',recipe=recipe,base_checkpoint=str(BASE),rewrite_checkpoint=str(REWRITER),arms={},limitations=['Single-seed small development pilot; labels are provisional assistant annotations, not human gold.','Validation conversations have been reused historically. Diagnostic conversations are previously inspected development material.','Reference-conditioned oracle route is diagnostic only, never a deployable result.','Soft probabilities condition decoder during both training and normal inference. Displayed hard labels are not the actual conditioning weights.','Auxiliary-only arm separates classification loss from decoder memory conditioning. No parameter-count or compute-equivalence claim.'])
    dump(OUT/'results.json',result)
    for name,(condition,auxiliary) in ARMS.items():
        torch.manual_seed(425);torch.cuda.manual_seed_all(425);random.seed(425)
        started=time.monotonic()
        m=ReplyActMT5(AutoModelForSeq2SeqLM.from_pretrained(BASE)).cuda()
        positives=torch.tensor([a for _,_,a in tr],device='cuda').sum(0)
        m.act_pos_weight.copy_(((len(tr)-positives)/positives.clamp_min(1)).clamp(1,5))
        m.model.gradient_checkpointing_enable();m.model.config.use_cache=False
        extras=list(m.classifier.parameters())+[m.act_embeddings]+list(m.act_norm.parameters())
        opt=torch.optim.AdamW([dict(params=m.model.parameters(),lr=3e-5),dict(params=extras,lr=3e-4)],weight_decay=.01)
        metrics=evaluate(m,va,tok,condition,auxiliary)
        arm=dict(history=[dict(epoch=0,validation=metrics)],best_epoch=0,best_ce=metrics['token_ce'],condition=condition,auxiliary=auxiliary,act_positive_weights=m.act_pos_weight.cpu().tolist(),outputs={})
        result['arms'][name]=arm
        checkpoint=OUT/name/'best.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save(m.state_dict(),checkpoint)
        stale=0
        for epoch in range(1,4):
            m.train();opt.zero_grad(set_to_none=True);batches=epoch_batches(len(tr),2,425,epoch);total=0.
            for bi,indices in enumerate(batches):
                # A partial accumulation group uses its actual size.
                group=min(4,len(batches)-(bi//4)*4)
                args=batch(tr,indices,tok)
                out=m(*args,condition=condition,auxiliary=auxiliary)
                if not torch.isfinite(out['loss']):raise FloatingPointError('Non-finite training loss')
                (out['loss']/group).backward();total+=float(out['loss'].detach())
                if (bi+1)%4==0 or bi+1==len(batches):
                    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();opt.zero_grad(set_to_none=True)
            metrics=evaluate(m,va,tok,condition,auxiliary)
            arm['history'].append(dict(epoch=epoch,train_batch_mean_loss=total/len(batches),validation=metrics))
            if metrics['token_ce']<arm['best_ce']:
                arm.update(best_ce=metrics['token_ce'],best_epoch=epoch);torch.save(m.state_dict(),checkpoint);stale=0
            else:stale+=1
            print(name,'epoch',epoch,'val',metrics,flush=True);dump(OUT/'results.json',result)
            if stale>=2:break
        del opt;gc.collect();torch.cuda.empty_cache()
        m.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));m.model.gradient_checkpointing_disable();m.model.config.use_cache=True
        arm['selected_validation']=evaluate(m,va,tok,condition,auxiliary)
        for split,data in [('validation',val),('diagnostic',diagnostics)]:
            arm['outputs'][split]={'predicted' if condition else 'none':generate(m,tok,data,'predicted' if condition else 'none')}
            if condition:arm['outputs'][split]['oracle']=generate(m,tok,data,'oracle')
        arm['training_and_standard_inference_seconds']=time.monotonic()-started
        dump(OUT/'results.json',result)
        del m;gc.collect();torch.cuda.empty_cache()
    # One unchanged rewriter for all arms and routes. No reference reply enters
    # the deployed-style predicted-act path, even for validation examples.
    local=V5Generator(REWRITER,device='cuda')
    for name,arm in result['arms'].items():
        for split,data in [('validation',val),('diagnostic',diagnostics)]:
            for route,outputs in arm['outputs'][split].items():
                anchors=[r['standard'] for r in outputs]
                for row,anchor in zip(data,anchors):
                    text=mt5_input('LOCAL_REWRITE',row['message'],row['context'],True,row['speakers'],standard_reply=anchor)
                    if len(local.tokenizer(text)['input_ids'])>local.max_source:raise ValueError('Rewrite source budget exceeded')
                texts=rewrite_standard(local.model,local.tokenizer,data,anchors,local.max_source,local.max_target,'cuda',DECODING,True)
                for item,anchor,text in zip(outputs,anchors,texts):item.update(local_standard_anchor=anchor,local=text)
                dump(OUT/'results.json',result)
    result['status']='complete';dump(OUT/'results.json',result)
    print('Completed matched mT5 reply-act pilot. No defaults changed.',flush=True)

if __name__=='__main__':main()
