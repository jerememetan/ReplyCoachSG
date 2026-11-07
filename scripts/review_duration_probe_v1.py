"""Anonymize all eight fixed snapshots; require complete explicit quality ratings."""
import argparse,hashlib,json,random,secrets
from collections import Counter
from pathlib import Path
STEPS=(160,320,480,784)
AXES=('standard','local','same_meaning','singlish')
def prepare(directory,source,output):
 if output.exists():raise FileExistsError(output)
 items=[json.loads(l) for l in source.read_text().splitlines()];ids={c['id'] for c in items}
 assert len(items)==len(ids)==24
 variants=[f'{arm}_step{step}' for arm in ('control','capacity') for step in STEPS]
 reports={name:json.loads((directory/f'{name}.json').read_text()) for name in variants}
 outputs={name:{r['item_id']:r for r in report['outputs']} for name,report in reports.items()}
 digest=hashlib.sha256(source.read_bytes()).hexdigest()
 for name,r in reports.items():
  assert r['fresh_source_sha256']==digest
  assert name==f"{r['arm']}_step{r['optimizer_step']}" and r['seed']==425
  assert len(r['outputs'])==24 and set(outputs[name])==ids
 rng=random.Random(secrets.randbits(64));questions=[];mapping={}
 for n,c in enumerate(sorted(items,key=lambda c:c['id']),1):
  order=variants[:];rng.shuffle(order);opts={};mapping[str(n)]={}
  for letter,name in zip('ABCDEFGH',order):
   r=outputs[name][c['id']];opts[letter]=dict(standard=r['generated_standard'],local=r['generated_local']);mapping[str(n)][letter]=name
  questions.append(dict(n=n,item_id=c['id'],conversation=[[t['speaker'],t['text']] for t in c['session_memory']],message=c['message']['text'],options=opts))
 output.mkdir(parents=True)
 packet=dict(rubric=dict(standard='Coherent next user turn; valid refusals/counterproposals and plausible selectable details allowed; explicit contradictions and role reversals fail.',local='Same contextual-adequacy criterion, independently judged.',same_meaning='Preserve essential actors, facts, choices, time, negation, certainty, requests and commitments.',singlish='Recognizably natural Singaporean phrasing/syntax; neutral-English copies and awkward particle stuffing do not pass.'),questions=questions,limitation='Assistant-authored development probe; not human gold. Review all fixed variants before opening key. Uncertain is not pass.')
 public=output/'packet.json';public.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
 (output/'key.json').write_text(json.dumps(dict(mapping=mapping,packet_sha256=hashlib.sha256(public.read_bytes()).hexdigest(),source_sha256={n:hashlib.sha256((directory/f'{n}.json').read_bytes()).hexdigest() for n in variants}),indent=2)+'\n')
def score(output):
 public=output/'packet.json';p=json.loads(public.read_text());k=json.loads((output/'key.json').read_text());r=json.loads((output/'ratings.json').read_text())
 assert k['packet_sha256']==r['packet_sha256']==hashlib.sha256(public.read_bytes()).hexdigest()
 expected={(q['n'],o) for q in p['questions'] for o in q['options']}
 assert len(r['ratings'])==len(expected) and {(x['n'],x['option']) for x in r['ratings']}==expected
 counts={}
 for row in r['ratings']:
  assert all(row.get(a) in ('pass','fail','uncertain') for a in AXES) and row.get('reason')
  name=k['mapping'][str(row['n'])][row['option']];c=counts.setdefault(name,Counter());c['pairs']+=1
  for a in AXES:c[a+'_'+row[a]]+=1
  c['all_pass']+=all(row[a]=='pass' for a in AXES)
 result=dict(counts=counts,packet_sha256=k['packet_sha256'],ratings_sha256=hashlib.sha256((output/'ratings.json').read_bytes()).hexdigest(),limitation='One seed, same24 assistant-authored development scenarios across8 checkpoint variants. Reviewer authored probe and knows previous behavior. Hidden labels do not create independent human blinding. Selecting an earlier checkpoint here is development selection, not final-test accuracy.')
 (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','score']);p.add_argument('--directory',type=Path);p.add_argument('--source',type=Path,default=Path('data/reward_diagnostics/duration_probe_v1.jsonl'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.action=='prepare':prepare(a.directory,a.source,a.output)
 else:score(a.output)
