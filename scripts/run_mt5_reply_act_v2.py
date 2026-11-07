"""Identity-initialized reply-act pilot with predicted-only and mixed supervision."""
import gc
import hashlib
import json
import random
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from replycoach.generate.reply_act_v2 import GatedReplyActMT5
from replycoach.generate.train_v5 import epoch_batches
from replycoach.generate.style_control import mt5_input
from replycoach.generate.v5_provider import V5Generator, DECODING
from replycoach.generate.rewrite_v5 import rewrite_standard
from run_mt5_reply_act_v1 import ROOT, BASE, REWRITER, OUT, read, dump, rows, encode, batch, evaluate, generate

ARMS = {'auxiliary_control': (False, False), 'gated_predicted': (True, False), 'gated_mixed': (True, True)}


def training_route(mixed, batch_index, epoch):
    # Alternate within accumulation groups; shift parity each epoch.
    return 'oracle' if mixed and (batch_index + epoch) % 2 == 0 else 'predicted'


def add_rewrites(result, local, data_by_split):
    for arm in result['arms'].values():
        for split,routes in arm['outputs'].items():
            for items in routes.values():
                eligible=[]
                for row,item in zip(data_by_split[split],items,strict=True):
                    if row['item_id']!=item['item_id']:raise ValueError('Rewrite IDs do not match')
                    anchor=item['standard']
                    if not anchor.strip():
                        item['rewrite_status']='blocked_empty_standard';continue
                    text=mt5_input('LOCAL_REWRITE',row['message'],row['context'],True,row['speakers'],standard_reply=anchor)
                    if len(local.tokenizer(text)['input_ids'])>local.max_source:
                        item['rewrite_status']='blocked_source_budget';continue
                    eligible.append((row,item))
                # Small batches also bound memory and permit progress saves.
                for start in range(0,len(eligible),4):
                    chunk=eligible[start:start+4]
                    anchors=[item['standard'] for _,item in chunk]
                    texts=rewrite_standard(local.model,local.tokenizer,[row for row,_ in chunk],anchors,local.max_source,local.max_target,'cuda',DECODING,True)
                    for (_,item),anchor,text in zip(chunk,anchors,texts,strict=True):
                        item.update(local_standard_anchor=anchor,local=text,rewrite_status='complete' if text.strip() else 'failed_empty_local')
                    dump(OUT/'results.json',result)


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise FileExistsError('Fresh experiment output required')
    recipe=read('recipe.json')
    for path,digest in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source mismatch: '+path)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    train,val=rows('train'),rows('validation')
    if {r['item_id'] for r in train}&{r['item_id'] for r in val}:raise ValueError('Train/validation overlap')
    oracle={r['item_id']:r['reply_acts'] for r in read('data/training_preferences/mt5_reply_act_v1/diagnostic_acts.json')}
    diagnostics=[{**r,'response':'','reply_acts':oracle[r['item_id']]} for r in read('data/evaluation/component_baseline_v1/inputs.json')]
    tok=AutoTokenizer.from_pretrained(BASE);tr=encode(train,tok);va=encode(val,tok)
    result=dict(status='running',recipe=recipe,base_checkpoint=str(BASE),rewrite_checkpoint=str(REWRITER),arms={},limitations=[
        'Single-seed development pilot with reused validation, previously inspected diagnostics and provisional assistant act annotations.',
        'Oracle targets are used only for training-split supervised batches and explicitly named validation/diagnostic oracle routes. Normal inference uses predictions only.',
        'gated_predicted never trains oracle conditioning; its oracle route remains distribution-shifted. gated_mixed explicitly trains both routes.',
        'Zero-initialized residual preserves original memory exactly; a small learned gate or unchanged replies may indicate conditioning is unused, not successful intent control.',
        'Validation token CE selects checkpoints; lower CE alone does not establish better contextual replies. No locked test access or app default changes.'])
    dump(OUT/'results.json',result)
    for name,(condition,mixed) in ARMS.items():
        torch.manual_seed(425);torch.cuda.manual_seed_all(425);random.seed(425)
        start=time.monotonic()
        m=GatedReplyActMT5(AutoModelForSeq2SeqLM.from_pretrained(BASE)).cuda()
        positives=torch.tensor([a for _,_,a in tr],device='cuda').sum(0)
        m.act_pos_weight.copy_(((len(tr)-positives)/positives.clamp_min(1)).clamp(1,5))
        m.model.gradient_checkpointing_enable();m.model.config.use_cache=False
        extras=list(m.classifier.parameters())+[m.act_embeddings]+list(m.act_norm.parameters())
        opt=torch.optim.AdamW([dict(params=m.model.parameters(),lr=3e-5),dict(params=extras,lr=3e-4),dict(params=[m.act_gate],lr=.01,weight_decay=0)],weight_decay=.01)
        initial=evaluate(m,va,tok,condition,True)
        original=evaluate(m,va,tok,False,True)
        gap=abs(initial['token_ce']-original['token_ce'])
        if gap>1e-7:raise ValueError('Identity initialization failed on actual checkpoint')
        arm=dict(condition=condition,mixed_training=mixed,initial_identity_ce_gap=gap,best_epoch=0,best_ce=initial['token_ce'],history=[dict(epoch=0,gate=0.,effective_residual_scale=0.,validation=initial)],training_batches={'predicted':0,'oracle':0},outputs={})
        result['arms'][name]=arm
        ckpt=OUT/name/'best.pt';ckpt.parent.mkdir(exist_ok=True,parents=True);torch.save(m.state_dict(),ckpt)
        stale=0
        for epoch in range(1,4):
            m.train();opt.zero_grad(set_to_none=True);batches=epoch_batches(len(tr),2,425,epoch);loss_sum=0.
            for bi,indices in enumerate(batches):
                route=training_route(mixed,bi,epoch)
                arm['training_batches'][route]+=1
                out=m(*batch(tr,indices,tok),condition=condition,auxiliary=True,training_route=route)
                if not torch.isfinite(out['loss']):raise FloatingPointError('Non-finite training loss')
                group=min(4,len(batches)-(bi//4)*4)
                (out['loss']/group).backward();loss_sum+=float(out['loss'].detach())
                if (bi+1)%4==0 or bi+1==len(batches):
                    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
            metrics=evaluate(m,va,tok,condition,True)
            history=dict(epoch=epoch,train_batch_mean_loss=loss_sum/len(batches),gate=float(m.act_gate.detach()),effective_residual_scale=float(.1*m.act_gate.detach().tanh()),validation=metrics)
            arm['history'].append(history)
            if metrics['token_ce']<arm['best_ce']:
                arm.update(best_ce=metrics['token_ce'],best_epoch=epoch);torch.save(m.state_dict(),ckpt);stale=0
            else:stale+=1
            dump(OUT/'results.json',result);print(name,history,flush=True)
            if stale>=2:break
        del opt;gc.collect();torch.cuda.empty_cache()
        m.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=True));m.model.gradient_checkpointing_disable();m.model.config.use_cache=True
        arm['selected_validation']=evaluate(m,va,tok,condition,True)
        arm['selected_gate']=float(m.act_gate.detach());arm['selected_effective_residual_scale']=float(.1*m.act_gate.detach().tanh())
        for split,data in [('validation',val),('diagnostic',diagnostics)]:
            routes=['none','predicted','oracle'] if condition else ['none']
            arm['outputs'][split]={route:generate(m,tok,data,route) for route in routes}
            for outputs in arm['outputs'][split].values():
                for item in outputs:item['standard_status']='complete' if item['standard'].strip() else 'failed_empty_standard'
        arm['training_and_standard_inference_seconds']=time.monotonic()-start
        dump(OUT/'results.json',result);del m;gc.collect();torch.cuda.empty_cache()
    local=V5Generator(REWRITER,device='cuda')
    add_rewrites(result,local,{'validation':val,'diagnostic':diagnostics})
    failures=[dict(arm=name,split=split,route=route,item_id=x['item_id'],status=x['rewrite_status']) for name,arm in result['arms'].items() for split,routes in arm['outputs'].items() for route,items in routes.items() for x in items if x['rewrite_status']!='complete']
    result.update(status='complete_with_failures' if failures else 'complete',failed_stages=failures)
    dump(OUT/'results.json',result);print('Completed gated mT5 reply-act pilot',result['status'],flush=True)

if __name__=='__main__':main()
