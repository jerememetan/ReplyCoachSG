"""Matched text-data workflow for mT5: public, dual task, rewrite and fidelity."""
import argparse,copy,gc,hashlib,json,random,traceback,time,platform
import transformers,peft
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoTokenizer
from replycoach.generate.mt5_pretrained_compat import OriginalMT5,original_config,verify_original_embeddings,original_tokenizer
from peft import LoraConfig,get_peft_model,PeftModel,get_peft_model_state_dict,set_peft_model_state_dict
from replycoach.generate.mt5_matched_v1 import source,encode,reply_loss,task_rows,paired_rows
from replycoach.generate.train_v5 import collate
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';REPORT={};TOKENIZER=None
DECODE={'greedy':dict(num_beams=1,do_sample=False,no_repeat_ngram_size=0),'beam4':dict(num_beams=4,do_sample=False,no_repeat_ngram_size=3)}
def read(p):return json.loads((ROOT/p).read_text())
def rows(p):return [json.loads(l) for l in (ROOT/p).read_text().splitlines() if l.strip()]
def save():
    OUT.mkdir(exist_ok=True);p=OUT/'results.tmp';p.write_text(json.dumps(REPORT,indent=2)+'\n');p.replace(OUT/'results.json')
def clear():gc.collect();torch.cuda.empty_cache()
def load(init=None):
    cfg=original_config(read('model/config.json'))
    base,info=OriginalMT5.from_pretrained(ROOT/'model',config=cfg,dtype=torch.bfloat16,local_files_only=True,output_loading_info=True)
    problems={k:list(info.get(k) or []) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']}
    if any(problems.values()):raise ValueError('Strict mT5 load failed: '+str(problems))
    verify_original_embeddings(base)
    base=base.cuda()
    if cfg.tie_word_embeddings!=read('model/config.json').get('tie_word_embeddings',False):raise ValueError('Unexpected tying policy')
    if init is None:model=get_peft_model(base,LoraConfig(task_type='SEQ_2_SEQ_LM',r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q','k','v','o','wi_0','wi_1','wo']))
    else:model=PeftModel.from_pretrained(base,init,is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    ps=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not ps or any('lora_' not in n for n,_ in ps):raise ValueError('Only LoRA parameters may train')
    REPORT['trainable_parameters']=sum(p.numel() for _,p in ps);REPORT['base_parameter_count']=sum(p.numel() for p in model.parameters())-REPORT['trainable_parameters'];return model

def tensors(encoded):
    ids,mask,labels=collate(encoded,TOKENIZER.pad_token_id);return ids.cuda(),mask.cuda(),labels.cuda()
@torch.no_grad()
def metrics(model,groups):
    model.eval();values={}
    for name,data in groups.items():
        total=0.;ntokens=0
        for start in range(0,len(data),2):
            ids,mask,labels=tensors(data[start:start+2]);out=model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False);n=int(labels.ne(-100).sum());total+=float(reply_loss(out.logits,labels))*n;ntokens+=n
        values[name]=dict(token_ce=total/ntokens,reply_tokens=ntokens,examples=len(data))
    return dict(groups=values,selection_score=sum(v['token_ce'] for v in values.values())/len(values))
def save_adapter(model,path):
    model.save_pretrained(path);TOKENIZER.save_pretrained(path)
    (path/'v5_input.json').write_text(json.dumps(dict(speaker_labels=True,joint=False,max_source=1536,max_target=64)))

def fit(name,train,groups,init,lr,accum,seed,public=False):
    phase=REPORT['phases'].get(name)
    if phase and phase.get('status')=='complete':return OUT/name/'best_adapter'
    if phase:REPORT.setdefault('previous_attempts',[]).append(dict(phase=name,report=copy.deepcopy(phase)))
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
    print('Starting phase',name,'examples',len(train),'initial_adapter',str(init),flush=True)
    model=load(init)
    if name.startswith('task_'):torch.manual_seed(seed+10000);torch.cuda.manual_seed_all(seed+10000)
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
    encoded=[encode(r,TOKENIZER,'rewrite' if 'standard_anchor' in r else 'direct') for r in train]
    valid={k:[encode(r,TOKENIZER,'rewrite' if 'standard_anchor' in r else 'direct') for r in rs] for k,rs in groups.items()}
    phase=dict(status='running',history=[],initial_validation=metrics(model,valid),train_exposures=len(train),validation_count=sum(map(len,valid.values())));REPORT['phases'][name]=phase
    path=OUT/name;path.mkdir(exist_ok=True);resume=path/'resume.pt';start=0;total=0.;best=None;stale=0
    if public and resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['recipe']!=REPORT['recipe'] or state['phase_name']!=name:raise ValueError('Resume recipe/phase mismatch')
        expected=get_peft_model_state_dict(model)
        if set(expected)!=set(state['adapter']):raise ValueError('Resume key mismatch')
        set_peft_model_state_dict(model,state['adapter'])
        if any(not torch.equal(v.detach().cpu(),state['adapter'][k]) for k,v in get_peft_model_state_dict(model).items()):raise ValueError('Resume adapter values differ')
        opt.load_state_dict(state['optimizer']);phase=state['phase'];REPORT['phases'][name]=phase;start=state['next_index'];total=state['total_loss'];torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
        if start%accum and start!=len(train):raise ValueError('Resume index not optimizer boundary')
    save()
    max_epochs=1 if public else 8
    for epoch in range(1,max_epochs+1):
        model.train();opt.zero_grad(set_to_none=True);order=torch.randperm(len(encoded),generator=torch.Generator().manual_seed(1425 if public else seed+epoch)).tolist()
        if not public:total=0.;start=0
        for i in range(start,len(order)):
            ids,mask,labels=tensors([encoded[order[i]]]);out=model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False);loss=reply_loss(out.logits,labels)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            group=min(accum,len(order)-(i//accum)*accum);(loss/group).backward();total+=float(loss.detach())
            if (i+1)%accum==0 or i+1==len(order):
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
                opt.step();opt.zero_grad(set_to_none=True)
                if i+1==min(accum,len(order)):
                    phase['first_update_finite']=True;save();print(name,'first optimizer update finite',flush=True)
            if public and ((i+1)%10000==0 or i+1==len(order)):
                scores=metrics(model,valid);phase['history'].append(dict(examples_seen=i+1,validation=scores));model.train();print(name,i+1,scores,flush=True)
            if public and ((i+1)%4000==0 or i+1==len(order)):
                phase.update(examples_seen=i+1,source_exposures=dict(Counter(train[j]['corpus'] for j in order[:i+1])))
                state=dict(recipe=REPORT['recipe'],phase_name=name,phase=phase,adapter={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()},optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),next_index=i+1,total_loss=total)
                tmp=path/'resume.tmp';torch.save(state,tmp);tmp.replace(resume);del state;save()
        if public:
            # A full-data intervention must retain the final complete-pass checkpoint.
            save_adapter(model,path/'best_adapter');phase['selection']='Final full public pass';phase['best_epoch']=1
        else:
            scores=metrics(model,valid);phase['history'].append(dict(epoch=epoch,train_example_mean_ce=total/len(encoded),validation=scores));print(name,epoch,scores,flush=True)
            if best is None or scores['selection_score']<best:best=scores['selection_score'];stale=0;phase.update(best_epoch=epoch,best_score=best);save_adapter(model,path/'best_adapter')
            else:stale+=1
        save()
        if not public and stale>=2:break
    phase.update(status='complete');save();del opt,params,model;clear();return path/'best_adapter'

@torch.no_grad()
def generate(model,data,style,method,anchors=None):
    model.eval();outputs=[]
    for i in range(0,len(data),4):
        batch=data[i:i+4];completed={};active=[];texts=[]
        for j,r in enumerate(batch):
            anchor=anchors[i+j] if anchors is not None else None
            if anchors is not None and not anchor.strip():
                completed[j]=dict(item_id=r['item_id'],text='',token_ids=[],status='failed_empty_anchor')
            else:
                active.append(j);texts.append(source(r,style,anchor))
        if active:
            inputs=TOKENIZER(texts,return_tensors='pt',padding=True)
            if inputs['input_ids'].shape[-1]>1536:raise ValueError('Generation source limit')
            seq=model.generate(**{k:v.cuda() for k,v in inputs.items()},max_new_tokens=64,use_cache=True,**DECODE[method])
            for j,ids in zip(active,seq,strict=True):
                text=TOKENIZER.decode(ids,skip_special_tokens=True);completed[j]=dict(item_id=batch[j]['item_id'],text=text,token_ids=ids.tolist(),status='failed_empty' if not text.strip() else 'complete' if TOKENIZER.eos_token_id in ids.tolist() else 'length_limit')
        outputs.extend(completed[j] for j in range(len(batch)))
    return outputs

def evaluate_direct(name,adapter,inputs):
    result=REPORT['phases'][name]
    if result.get('outputs_complete'):return
    model=load(adapter).eval();result['outputs']={}
    for method in DECODE:
        english=generate(model,inputs,'STANDARD',method);local=generate(model,inputs,'LOCAL',method)
        result['outputs'][method]=[dict(item_id=a['item_id'],standard=a,local=b) for a,b in zip(english,local,strict=True)];save()
    result['outputs_complete']=True;save();del model;clear()

def evaluate_rewrite(name,adapter,inputs,own_english,challenge=None):
    result=REPORT['phases'][name]
    if result.get('outputs_complete'):return
    model=load(adapter).eval();result['outputs']={}
    for method in DECODE:
        routes=[('actual_mt5',inputs,[own_english[method][r['item_id']] for r in inputs]),('shared_sea_english',inputs,[r['actual_anchor'] for r in inputs])]
        good=[r for r in inputs if 'diagnostic_anchor' in r];routes.append(('diagnostic',good,[r['diagnostic_anchor'] for r in good]))
        if challenge:routes.append(('challenge',challenge,[r['standard_anchor'] for r in challenge]))
        result['outputs'][method]={}
        for route,cases,anchors in routes:
            results=generate(model,cases,'LOCAL_REWRITE',method,anchors)
            result['outputs'][method][route]=[dict(**x,standard_anchor=a) for x,a in zip(results,anchors,strict=True)];save()
    result['outputs_complete']=True;save();del model;clear()

def main():
    global REPORT,TOKENIZER
    ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['small','full'],required=True);args=ap.parse_args();recipe=read('recipe.json')
    if args.branch!=recipe['branch']:raise ValueError('Branch/recipe mismatch')
    for p,h in recipe['files_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Source hash mismatch '+p)
    bm=read('base_manifest.json')
    for p,h in bm['saved_files_sha256'].items():
        if hashlib.sha256((ROOT/'model'/p).read_bytes()).hexdigest()!=h:raise ValueError('Base checkpoint changed')
    if not (ROOT/'data/training_expansion/generated600_repair_v1/combined_train.jsonl').read_bytes().startswith((ROOT/'data/dev/train.jsonl').read_bytes()):raise ValueError('Team records changed')
    torch.set_num_threads(4);TOKENIZER=original_tokenizer(ROOT/'model')
    if 'L40S' not in torch.cuda.get_device_name(0):raise ValueError('L40S required')
    REPORT=read('artifacts/results.json') if (OUT/'results.json').exists() else dict(status='running',recipe=recipe,base_manifest=bm,phases={},default_changed=False)
    if REPORT['recipe']!=recipe:raise ValueError('Run recipe changed')
    if REPORT.get('status') in ['complete','complete_with_flags']:raise FileExistsError('Complete experiment')
    REPORT['environment']=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,gpu=torch.cuda.get_device_name(0));REPORT['status']='running';REPORT.pop('error',None);REPORT.pop('traceback',None);save()
    try:
        pubpath='data/public/sealion_dialogue_v1' if args.branch=='small' else 'data/public/sealion_full_v1'
        public=rows(pubpath+'/train.jsonl');pv=rows(pubpath+'/validation.jsonl')
        if {r['group_id'] for r in public}&{r['group_id'] for r in pv}:raise ValueError('Public group overlap')
        pubvalid={c:[r for r in pv if r['corpus']==c] for c in ['cosem','dailydialog']}
        public_adapter=fit('public',public,pubvalid,None,5e-5,16,425,True)
        original=rows('data/training_expansion/generated600_repair_v1/combined_train.jsonl');ov=rows('data/dev/validation.jsonl');task=task_rows(original);val=task_rows(ov)
        if len(task)!=1580 or {r['item_id'] for r in task}&{r['item_id'] for r in val}:raise ValueError('Task counts or overlap')
        valid={s:[r for r in val if r['style']==s] for s in ['STANDARD','LOCAL']};evaluation=read('data/evaluation/sealion_rewrite_v1/inputs.json')
        adapters={}
        for seed in ([425,426] if args.branch=='small' else [425]):
            adapters[seed]=fit(f'task_s{seed}',task,valid,public_adapter,1e-4,8,seed)
            evaluate_direct(f'task_s{seed}',adapters[seed],evaluation)
        if args.branch=='small':
            audit=read('data/training_expansion/sealion_rewrite_v1/pair_audit.json');tr={r['id'] for r in audit['items'] if r['split']=='train' and r['eligible']};va={r['id'] for r in audit['items'] if r['split']=='validation' and r['eligible']}
            replay=paired_rows(original,tr);rv=paired_rows(ov,va)
            own={m:{x['item_id']:x['standard']['text'] for x in REPORT['phases']['task_s425']['outputs'][m]} for m in DECODE}
            rewrite=fit('rewrite',replay,{'REWRITE':rv},adapters[425],3e-5,8,425)
            challenge=read('data/training_expansion/sealion_fidelity_v1/challenge.json')
            evaluate_rewrite('rewrite',rewrite,evaluation,own,challenge)
            target=read('data/training_expansion/sealion_fidelity_v1/train.json');fv=read('data/training_expansion/sealion_fidelity_v1/validation.json');fidelity=replay+[target[i%len(target)] for i in range(len(replay))]
            fa=fit('fidelity',fidelity,{'REPLAY':rv,'FIDELITY':fv},rewrite,3e-5,8,425)
            evaluate_rewrite('fidelity',fa,evaluation,own,challenge)
        flags=[]
        def inspect(obj,path):
            if isinstance(obj,dict):
                if obj.get('status') in ['failed_empty','failed_empty_anchor','length_limit']:flags.append(dict(path=path,status=obj['status'],item_id=obj.get('item_id')))
                for key,value in obj.items():inspect(value,path+'/'+str(key))
            elif isinstance(obj,list):
                for i,value in enumerate(obj):
                    if isinstance(value,(dict,list)):inspect(value,path+'/'+str(i))
        inspect(REPORT['phases'],'phases')
        REPORT.update(status='complete_with_flags' if flags else 'complete',flagged_outputs=flags);save()
    except Exception as exc:
        REPORT.update(status='failed',error=str(exc),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
