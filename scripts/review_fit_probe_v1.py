"""Review STANDARD fit and wording transfer, with exact-target matches identified separately."""
import argparse,hashlib,json,random,secrets
from collections import Counter
from pathlib import Path
from replycoach.generate.fit_probe_v1 import normalize_reply
VARIANTS=('base','lora_lr1e4','lora_lr3e4','full_lr1e4','full_lr3e4')
def prepare(directory,data,output):
 if output.exists():raise FileExistsError(output)
 reports={name:json.loads((directory/f'{name}.json').read_text()) for name in VARIANTS}
 for name,r in reports.items():
  if name!='base':assert r['status']=='complete' and r['optimizer_steps']==400 and r['train_items']==24
 assert all(r['base_files_sha256']==reports['base']['base_files_sha256'] for r in reports.values())
 rng=random.Random(secrets.randbits(64));questions=[];mapping={};prefill=[]
 for split,file in [('train','train.jsonl'),('paraphrase','paraphrases.jsonl')]:
  items=[json.loads(l) for l in (data/file).read_text().splitlines()];assert len(items)==24
  runs={name:{x['item_id']:x for x in (r['train_generations'] if name=='base' else r['train_generations']['400'])} if split=='train' else {x['item_id']:x for x in r['paraphrase_generations']} for name,r in reports.items()}
  assert all(set(run)=={c['item_id'] for c in items} for run in runs.values())
  for c in sorted(items,key=lambda c:c['item_id']):
   n=len(questions)+1;order=list(VARIANTS);rng.shuffle(order);options={};mapping[str(n)]={}
   for o,name in zip('ABCDE',order):
    row=runs[name][c['item_id']];text=row['generated_standard'];exact=normalize_reply(text)==normalize_reply(c['response']);assert row['exact_target_match']==exact
    options[o]=text;mapping[str(n)][o]=name
    if exact:prefill.append(dict(n=n,option=o,adequacy='pass',reason='Exact normalized match to the prechecked STANDARD target; automatic target-adherence annotation, not a new human judgment.',rating_source='prechecked exact-target match'))
   questions.append(dict(n=n,split=split,item_id=c['item_id'],conversation=list(zip(c['speakers'],c['context'])),message=c['message'],options=options))
 output.mkdir(parents=True);public=output/'packet.json'
 public.write_text(json.dumps(dict(questions=questions,rubric='Coherent usable next user turn respecting explicit roles, facts and constraints. Valid alternatives/refusals/counterproposals and plausible selectable details allowed. Pass/fail/uncertain; nonexact wording is not automatically wrong. No LOCAL/style-retention assessment.',limitation='One seed;24 underlying scenarios repeated in paraphrases. Reviewer checked/authored targets. This is a fit diagnostic, not human accuracy or an independently blind final evaluation.'),ensure_ascii=False,indent=2)+'\n')
 digest=hashlib.sha256(public.read_bytes()).hexdigest()
 (output/'key.json').write_text(json.dumps(dict(mapping=mapping,packet_sha256=digest,source_sha256={n:hashlib.sha256((directory/f'{n}.json').read_bytes()).hexdigest() for n in reports}),indent=2)+'\n')
 (output/'ratings.json').write_text(json.dumps(dict(packet_sha256=digest,ratings=prefill,limitation='Exact-target match annotations are separated from assistant inspection of alternative generations.'),indent=2)+'\n')
 print('Prepared240 replies; exact-target annotations:',len(prefill),'; remaining:',240-len(prefill))
def score(output):
 public=output/'packet.json';p=json.loads(public.read_text());k=json.loads((output/'key.json').read_text());r=json.loads((output/'ratings.json').read_text())
 assert k['packet_sha256']==r['packet_sha256']==hashlib.sha256(public.read_bytes()).hexdigest()
 expected={(q['n'],o) for q in p['questions'] for o in q['options']};assert len(r['ratings'])==len(expected) and {(x['n'],x['option']) for x in r['ratings']}==expected
 splits={q['n']:q['split'] for q in p['questions']};counts={}
 for row in r['ratings']:
  assert row['adequacy'] in ('pass','fail','uncertain') and row.get('reason')
  name=k['mapping'][str(row['n'])][row['option']];c=counts.setdefault(splits[row['n']],{}).setdefault(name,Counter());c['cases']+=1;c[row['adequacy']]+=1
  c['exact_target_annotations']+=row.get('rating_source')=='prechecked exact-target match'
 result=dict(counts=counts,training_fit_gate=22,training_gate_pass={name:c.get('pass',0)>=22 for name,c in counts['train'].items() if name!='base'},packet_sha256=k['packet_sha256'],ratings_sha256=hashlib.sha256((output/'ratings.json').read_bytes()).hexdigest(),limitation='STANDARD-only fit diagnostic. Automatic exact-target adherence plus assistant inspection of alternatives.24 underlying scenarios, not48 independent scenarios. No human-accuracy or production-quality claim.')
 (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','score']);p.add_argument('--directory',type=Path);p.add_argument('--data',type=Path,default=Path('data/training_diagnostics/fit_probe_v1'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.action=='prepare':prepare(a.directory,a.data,a.output)
 else:score(a.output)
