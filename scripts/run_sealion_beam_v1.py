"""Compare frozen SEA direct/English-first pipelines under greedy and beam search."""
import copy,gc,hashlib,json,time,traceback
from pathlib import Path
import torch
from peft import PeftModel
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
from replycoach.generate.sealion_data import source
from replycoach.generate.sealion_dual_v1 import generate_reply,BOUNDARY as DIRECT_BOUNDARY
from replycoach.generate.sealion_rewrite_v1 import generate_rewrite,rewrite_source,BOUNDARY as REWRITE_BOUNDARY
from replycoach.generate.sealion_beam_v1 import beam_decode
from run_sealion_dual_v1 import ROOT,OUT,read,save_report,load_base
METHODS={'greedy':None,'beam4':dict(num_beams=4,no_repeat_ngram_size=0),
         'beam8':dict(num_beams=8,no_repeat_ngram_size=0),'beam4_ng3':dict(num_beams=4,no_repeat_ngram_size=3)}

def main():
 if (OUT/'results.json').exists():raise FileExistsError('Fresh snapshot required')
 recipe=read('recipe.json');download=read('download_manifest.json')
 if download['revision']!=recipe['revision'] or download['weights_sha256']!=recipe['weights_sha256']:raise ValueError('Base identity mismatch')
 for p,h in recipe['files_sha256'].items():
  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Frozen source mismatch '+p)
 def verify_adapters():
  for folder,files in recipe['adapters_sha256'].items():
   for name,h in files.items():
    if hashlib.sha256((ROOT/folder/name).read_bytes()).hexdigest()!=h:raise ValueError('Adapter identity changed')
 verify_adapters();torch.set_num_threads(4)
 if 'L40S' not in torch.cuda.get_device_name(0):raise ValueError('L40S required')
 reused=read('data/evaluation/sealion_full_rewrite_v1/inputs.json');fresh=read('data/evaluation/restored_pipeline_v1/inputs.json');cases=reused+fresh
 challenge=read('data/training_expansion/sealion_fidelity_v1/challenge.json')
 baseline=read('artifacts/sealion_full_v1/results.json');fixed_rewriter=read('artifacts/sealion_full_rewrite_v1/results.json')
 b={x['item_id']:x for rs in baseline['outputs'].values() for x in rs}
 tok=checkpoint_tokenizer(ROOT/'model');report=dict(status='running',recipe=recipe,methods={},default_changed=False,training_performed=False,greedy_replay_verified=dict(direct=0,rewrite=0));save_report(report)
 started=time.monotonic()
 try:
  base,problems=load_base();report['strict_loading_problems']=problems;model=PeftModel.from_pretrained(base,ROOT/'direct_adapter',is_trainable=False).eval()
  for method,args in METHODS.items():
   result=dict(direct=[],rewrite={});report['methods'][method]=result;save_report(report);begin=time.monotonic()
   for row in cases:
    item=dict(item_id=row['item_id'],styles={})
    for style in ['STANDARD','LOCAL']:
     if args is None:out=generate_reply(model,tok,row,style)
     else:
      ids=tok.encode(source(row,style),add_special_tokens=False)
      out=beam_decode(model,tok,ids,DIRECT_BOUNDARY,length_penalty=1.,**args)
     item['styles'][style]=out
     if method=='greedy' and row['item_id'] in b:
      expected=b[row['item_id']]['styles'][style]
      if out['reply']!=expected['reply'] or out['termination']!=expected['termination']:raise ValueError('Greedy direct replay changed')
      report['greedy_replay_verified']['direct']+=1
    result['direct'].append(item);save_report(report)
   result['direct_seconds']=time.monotonic()-begin;print(method,'direct generation finished',flush=True)
  del model,base;gc.collect();torch.cuda.empty_cache()
  base,problems=load_base();model=PeftModel.from_pretrained(base,ROOT/'rewrite_adapter',is_trainable=False).eval();cache={}
  for method,args in METHODS.items():
   direct={x['item_id']:x for x in report['methods'][method]['direct']}
   diagnostics=[r for r in reused if 'diagnostic_anchor' in r]
   groups=[('fixed_english',reused,[r['actual_anchor'] for r in reused]),
           ('own_english',cases,[direct[r['item_id']]['styles']['STANDARD']['reply'] for r in cases]),
           ('diagnostic',diagnostics,[r['diagnostic_anchor'] for r in diagnostics]),
           ('challenge',challenge,[r['standard_anchor'] for r in challenge]),
           ('fresh_diagnostic',fresh,[r['diagnostic_anchor'] for r in fresh])]
   begin=time.monotonic();report['methods'][method]['rewrite_cache_hits']=0
   for route,rows,anchors in groups:
    results=[];report['methods'][method]['rewrite'][route]=results
    for row,anchor in zip(rows,anchors,strict=True):
     key=(method,row['item_id'],anchor)
     if not anchor.strip():out=dict(reply='',standard_anchor=anchor,status='failed_empty_anchor')
     elif key in cache:out=copy.deepcopy(cache[key]);report['methods'][method]['rewrite_cache_hits']+=1
     else:
      if args is None:out=generate_rewrite(model,tok,row,anchor)
      else:
       ids=tok.encode(rewrite_source(row,anchor),add_special_tokens=False)
       out=beam_decode(model,tok,ids,REWRITE_BOUNDARY,length_penalty=1.,**args);out['standard_anchor']=anchor
      cache[key]=copy.deepcopy(out)
     if out['standard_anchor']!=anchor:raise ValueError('English anchor changed')
     if method=='greedy' and route in ['fixed_english','diagnostic','challenge']:
      original_route={'fixed_english':'actual_english','diagnostic':'diagnostic_english','challenge':'controlled_challenge'}[route]
      expected=next(x['rewrite'] for x in fixed_rewriter['outputs'][original_route] if x['item_id']==row['item_id'])
      if out['reply']!=expected['reply'] or out.get('termination')!=expected['termination']:raise ValueError('Greedy rewrite replay changed')
      report['greedy_replay_verified']['rewrite']+=1
     results.append(dict(item_id=row['item_id'],standard_anchor=anchor,rewrite=out));save_report(report)
   report['methods'][method]['rewrite_seconds']=time.monotonic()-begin;print(method,'rewrite generation finished',flush=True)
  verify_adapters()
  if report['greedy_replay_verified']!=dict(direct=164,rewrite=132):raise ValueError('Incomplete greedy replay checks')
  flags=[]
  for method,result in report['methods'].items():
   for item in result['direct']:
    for style,out in item['styles'].items():
     if out['status']!='complete':flags.append(dict(method=method,route='direct',style=style,item_id=item['item_id'],status=out['status']))
   for route,rs in result['rewrite'].items():
    for item in rs:
     if item['rewrite']['status']!='complete':flags.append(dict(method=method,route=route,item_id=item['item_id'],status=item['rewrite']['status']))
  report.update(status='complete_with_flags' if flags else 'complete',flagged_outputs=flags,adapters_unchanged=True,seconds=time.monotonic()-started);save_report(report)
 except Exception as exc:report.update(status='failed',error=str(exc),traceback=traceback.format_exc());save_report(report);raise
if __name__=='__main__':main()
