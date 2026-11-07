"""Paired-goal mT5 pilot. Primary question: can a supplied act control the reply?"""
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from replycoach.generate.act_follow_v1 import (ActFollowMT5,conversation_source,conditioned_source,
    supported_act_patterns,choose_predicted_acts)
from replycoach.generate.reply_act_v1 import REPLY_ACTS,act_vector
from replycoach.generate.train_v5 import collate,epoch_batches
from replycoach.generate.v5_provider import V5Generator,DECODING
from replycoach.generate.rewrite_v5 import rewrite_standard
from replycoach.generate.style_control import mt5_input
from run_mt5_reply_act_v1 import ROOT,BASE,REWRITER,OUT,read,dump,rows

DATA='data/training_expansion/mt5_act_follow_v1'
MODES={'direct_control':'direct','gated_paired':'gated','text_goal_paired':'text'}


def selected_acts(row,route,predicted):
    if route not in {'supplied','predicted','disabled'}:raise ValueError(route)
    # The normal route must remain independent of evaluation annotations.
    return row['reply_acts'] if route=='supplied' else predicted


def attach_pairs(data):
    groups={}
    for r in data:groups.setdefault(r['conversation_id'],[]).append(r)
    result=[]
    for group in groups.values():
        if len(group)!=2:raise ValueError('Expected two reply goals per conversation')
        for n,r in enumerate(group):
            result.append({**r,'is_original':False,'wrong_acts':group[1-n]['reply_acts']})
    return result


def original(split):
    return [{**r,'is_original':True,'wrong_acts':r['reply_acts']} for r in rows(split)]


def encode(data,tok):
    result=[]
    for r in data:
        src=tok(conversation_source(r))['input_ids']
        yes=tok(conditioned_source(r,r['reply_acts']))['input_ids']
        no=tok(conditioned_source(r,r['wrong_acts']))['input_ids']
        target=tok(r['response'])['input_ids']
        if len(src)>192 or max(len(yes),len(no))>256 or len(target)>48:
            raise ValueError(f"No truncation allowed: {r['item_id']} sizes {len(src)}/{len(yes)}/{len(no)}/{len(target)}")
        result.append(dict(source=src,conditioned=yes,wrong=no,target=target,
            acts=act_vector(r['reply_acts']),wrong_acts=act_vector(r['wrong_acts']),original=r['is_original']))
    return result


def batch(data,indices,tok):
    rs=[data[i] for i in indices]
    ids,mask,labels=collate([(r['source'],r['target']) for r in rs],tok.pad_token_id)
    def pad(key):
        tensor=pad_sequence([torch.tensor(r[key]) for r in rs],batch_first=True,padding_value=tok.pad_token_id).cuda()
        return tensor,tensor.ne(tok.pad_token_id)
    yes,yesmask=pad('conditioned');no,nomask=pad('wrong')
    return dict(input_ids=ids.cuda(),attention_mask=mask.cuda(),labels=labels.cuda(),
        acts=torch.tensor([r['acts'] for r in rs],device='cuda'),
        original_mask=torch.tensor([r['original'] for r in rs],device='cuda'),
        conditioned_ids=yes,conditioned_mask=yesmask,
        wrong_acts=torch.tensor([r['wrong_acts'] for r in rs],device='cuda'),
        wrong_ids=no,wrong_mask=nomask,paired_mask=torch.tensor([not r['original'] for r in rs],device='cuda'))


@torch.no_grad()
def evaluate(m,data,tok):
    m.eval();loss=0.;count=0;rank=0.;pair_count=0;truth=[];pred=[]
    for start in range(0,len(data),2):
        indices=list(range(start,min(start+2,len(data))))
        args=batch(data,indices,tok);out=m(**args)
        tokens=int(args['labels'].ne(-100).sum());loss+=float(out['token_loss'])*tokens;count+=tokens
        pairs=int(args['paired_mask'].sum());rank+=float(out['rank_loss'])*pairs;pair_count+=pairs
        if bool(args['original_mask'].any()):
            idx=args['original_mask'].bool()
            truth.append(args['acts'][idx].bool().cpu());pred.append(out['act_logits'][idx].sigmoid().ge(.5).cpu())
    result=dict(supplied_goal_token_ce=loss/count,paired_ranking_hinge=rank/max(pair_count,1))
    if truth:
        t,p=torch.cat(truth),torch.cat(pred);tp=(t&p).sum();fp=(p&~t).sum();fn=(t&~p).sum()
        result.update(classifier_micro_f1=float(2*tp/(2*tp+fp+fn).clamp_min(1)),classifier_exact_match=float((t==p).all(-1).float().mean()))
    return result


@torch.no_grad()
def render(m,tok,data,route,patterns):
    if route not in {'supplied','predicted','disabled'}:raise ValueError(route)
    m.eval();outputs=[]
    for r in data:
        ids=tok(conversation_source(r),return_tensors='pt')['input_ids'].cuda();mask=torch.ones_like(ids)
        if ids.shape[-1]>192:raise ValueError('Source exceeds inference budget')
        base,actual,logits=m.memory(ids,mask,'none')
        predicted=choose_predicted_acts(logits,patterns)[0]
        chosen=selected_acts(r,route,predicted)
        acts=torch.tensor([act_vector(chosen)],device='cuda')
        cid=tok(conditioned_source(r,chosen),return_tensors='pt')['input_ids'].cuda()
        if cid.shape[-1]>256:raise ValueError('Conditioned source exceeds inference budget')
        encoded,actual=m.render_memory(base,actual,acts,cid,torch.ones_like(cid),disabled=route=='disabled')
        generated=m.model.generate(encoder_outputs=encoded,attention_mask=actual,max_new_tokens=48,**DECODING)
        reply=tok.decode(generated[0],skip_special_tokens=True)
        outputs.append(dict(item_id=r['item_id'],conversation_id=r.get('conversation_id',r['item_id']),
            route=route,selected_acts=chosen,predicted_acts=predicted,
            selected_acts_are_privileged=route=='supplied',standard=reply,
            standard_status='complete' if reply.strip() else 'failed_empty_standard'))
    return outputs


def add_rewrites(result,data_map):
    local=V5Generator(REWRITER,device='cuda')
    for arm in result['arms'].values():
        for name,outputs in arm['outputs'].items():
            eligible=[]
            for row,item in zip(data_map[name],outputs,strict=True):
                if row['item_id']!=item['item_id']:raise ValueError('Rewrite ID mismatch')
                anchor=item['standard']
                if not anchor.strip():item['rewrite_status']='blocked_empty_standard';continue
                text=mt5_input('LOCAL_REWRITE',row['message'],row['context'],True,row['speakers'],standard_reply=anchor)
                if len(local.tokenizer(text)['input_ids'])>local.max_source:
                    item['rewrite_status']='blocked_source_budget';continue
                eligible.append((row,item))
            for start in range(0,len(eligible),4):
                chunk=eligible[start:start+4];anchors=[item['standard'] for _,item in chunk]
                replies=rewrite_standard(local.model,local.tokenizer,[row for row,_ in chunk],anchors,local.max_source,local.max_target,'cuda',DECODING,True)
                for (_,item),anchor,text in zip(chunk,anchors,replies,strict=True):
                    item.update(local_standard_anchor=anchor,local=text,rewrite_status='complete' if text.strip() else 'failed_empty_local')
                dump(OUT/'results.json',result)


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise FileExistsError('Fresh report required')
    recipe=read('recipe.json')
    for p,h in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Frozen file changed: '+p)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    oldtrain,oldval=original('train'),original('validation')
    pairedtrain=attach_pairs(read(DATA+'/train.json'));pairedval=attach_pairs(read(DATA+'/validation.json'))
    evaluation=attach_pairs(read(DATA+'/evaluation.json'))
    train=oldtrain+pairedtrain;patterns=supported_act_patterns(train)
    # One automatic reply per conversation. Variant selection here only supplies
    # the common context; render(predicted) never reads its reference acts/reply.
    auto=evaluation[::2]
    data_map=dict(supplied_evaluation=evaluation,disabled_evaluation=evaluation,
        predicted_evaluation=auto,predicted_original_validation=oldval)
    tok=AutoTokenizer.from_pretrained(BASE)
    tr=encode(train,tok);vo=encode(oldval,tok);vp=encode(pairedval,tok)
    result=dict(status='running',recipe=recipe,base_checkpoint=str(BASE),rewrite_checkpoint=str(REWRITER),
        supported_act_patterns=patterns,arms={},limitations=[
            'New paired targets, validation and evaluation are assistant-authored, not independently human-reviewed gold.',
            '16 fresh evaluation conversation groups are frozen before training/first output inspection; similar structures occur in training, so this measures limited structural generalization.',
            'Classifier is supervised only on original 190 provisional labels. Paired alternative goals are not treated as uniquely predictable.',
            'Supplied-goal evaluation is privileged controllability, not end-to-end autonomous quality. Predicted route is separate.',
            'Direct control uses identical reply targets without act input; differing legitimate targets make exact text agreement unsuitable for contextual quality scoring.',
            'Conditional arms also use paired ranking loss, so this does not isolate interface from objective or match compute.',
            'Checkpoint selection uses supplied-goal validation CE, not evaluation outputs. No locked product test access or default changes.'])
    dump(OUT/'results.json',result)
    for name,mode in MODES.items():
        torch.manual_seed(425);torch.cuda.manual_seed_all(425);random.seed(425);started=time.monotonic()
        m=ActFollowMT5(AutoModelForSeq2SeqLM.from_pretrained(BASE),mode).cuda()
        positives=torch.tensor([act_vector(r['reply_acts']) for r in oldtrain],device='cuda').sum(0)
        m.act_pos_weight.copy_(((len(oldtrain)-positives)/positives.clamp_min(1)).clamp(1,5))
        m.model.gradient_checkpointing_enable();m.model.config.use_cache=False
        extras=list(m.classifier.parameters())+[m.act_embeddings]+list(m.act_norm.parameters())
        opt=torch.optim.AdamW([dict(params=m.model.parameters(),lr=3e-5),dict(params=extras,lr=3e-4),dict(params=[m.act_gate],lr=.01,weight_decay=0)],weight_decay=.01)
        def metrics():
            a,b=evaluate(m,vo,tok),evaluate(m,vp,tok)
            return dict(original_validation=a,paired_validation=b,
                selection_score=.5*(a['supplied_goal_token_ce']+b['supplied_goal_token_ce']),
                effective_residual_scale=float(.1*m.act_gate.detach().tanh()))
        initial=metrics();arm=dict(mode=mode,best_epoch=0,best_score=initial['selection_score'],history=[dict(epoch=0,**initial)],outputs={})
        result['arms'][name]=arm;ckpt=OUT/name/'best.pt';ckpt.parent.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),ckpt)
        stale=0
        for epoch in range(1,7):
            m.train();opt.zero_grad(set_to_none=True);batches=epoch_batches(len(tr),2,425,epoch);tot=0.
            for bi,indices in enumerate(batches):
                out=m(**batch(tr,indices,tok))
                if not torch.isfinite(out['loss']):raise FloatingPointError('Non-finite training loss')
                group=min(4,len(batches)-(bi//4)*4);(out['loss']/group).backward();tot+=float(out['loss'].detach())
                if (bi+1)%4==0 or bi+1==len(batches):
                    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
            val=metrics();arm['history'].append(dict(epoch=epoch,train_batch_mean_loss=tot/len(batches),**val))
            if val['selection_score']<arm['best_score']:
                arm.update(best_score=val['selection_score'],best_epoch=epoch);torch.save(m.state_dict(),ckpt);stale=0
            else:stale+=1
            dump(OUT/'results.json',result);print(name,epoch,val,flush=True)
            if stale>=2:break
        del opt;gc.collect();torch.cuda.empty_cache()
        m.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=True));m.model.gradient_checkpointing_disable();m.model.config.use_cache=True
        arm['selected_validation']=metrics()
        for key,data in data_map.items():
            route='supplied' if key.startswith('supplied') else 'disabled' if key.startswith('disabled') else 'predicted'
            arm['outputs'][key]=render(m,tok,data,route,patterns);dump(OUT/'results.json',result)
        arm['training_and_standard_seconds']=time.monotonic()-started
        del m;gc.collect();torch.cuda.empty_cache()
    add_rewrites(result,data_map)
    failed=[dict(arm=n,route=route,item_id=x['item_id'],status=x['rewrite_status']) for n,a in result['arms'].items() for route,xs in a['outputs'].items() for x in xs if x['rewrite_status']!='complete']
    result.update(status='complete_with_failures' if failed else 'complete',failed_stages=failed)
    dump(OUT/'results.json',result);print('Completed paired-goal mT5 pilot',result['status'],flush=True)

if __name__=='__main__':main()
