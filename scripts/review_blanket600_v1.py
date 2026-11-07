"""Review final matched-budget control/expanded outputs on reused development contexts."""
import argparse,hashlib,json,random,secrets
from collections import Counter
from pathlib import Path
AXES=('standard','local','same_meaning','singlish')
def prepare(directory,output):
 if output.exists():raise FileExistsError(output)
 reports={a:json.loads((directory/f'{a}_final_outputs.json').read_text()) for a in ['control','expanded']}
 assert reports['control']['optimizer_steps']==reports['expanded']['optimizer_steps']
 assert reports['control']['preflight']==reports['expanded']['preflight']
 rng=random.Random(secrets.randbits(64));questions=[];mapping={}
 for split,path,field in [('validation',Path('data/dev/validation.jsonl'),'validation_outputs'),('diagnostic',Path('data/reward_diagnostics/duration_probe_v1.jsonl'),'diagnostic_outputs')]:
  items=[json.loads(l) for l in path.read_text().splitlines()];assert len(items)==(50 if split=='validation' else 24)
  runs={a:{r['item_id']:r for r in report[field]} for a,report in reports.items()}
  assert all(len(report[field])==len(items) and set(runs[a])=={c['id'] for c in items} for a,report in reports.items())
  for c in sorted(items,key=lambda x:x['id']):
   n=len(questions)+1;order=['control','expanded'];rng.shuffle(order);options={};mapping[str(n)]={}
   for letter,arm in zip('AB',order):
    r=runs[arm][c['id']];options[letter]=dict(standard=r['generated_standard'],local=r['generated_local']);mapping[str(n)][letter]=arm
   questions.append(dict(n=n,split=split,item_id=c['id'],conversation=[[t['speaker'],t['text']] for t in c['session_memory']],message=c['message']['text'],options=options))
 output.mkdir(parents=True);public=output/'packet.json'
 public.write_text(json.dumps(dict(questions=questions,rubric=dict(standard='Coherent next user turn; valid refusals/counterproposals and plausible selectable details allowed; explicit contradictions/role reversals fail.',local='Same contextual adequacy, separately judged.',same_meaning='Preserve essential actors, facts, time, decisions, requests, negation and commitments.',singlish='Recognizably natural Singlish; neutral English copies and awkward particle stuffing do not pass.'),limitation='Reused50+24 development conversations; assistant authored/curated training examples and reviewed prior models. Labels hidden but identities may be inferred. Not independent human gold.'),ensure_ascii=False,indent=2)+'\n')
 (output/'key.json').write_text(json.dumps(dict(mapping=mapping,packet_sha256=hashlib.sha256(public.read_bytes()).hexdigest(),source_sha256={a:hashlib.sha256((directory/f'{a}_final_outputs.json').read_bytes()).hexdigest() for a in reports}),indent=2)+'\n')
def score(output):
 p=json.loads((output/'packet.json').read_text());k=json.loads((output/'key.json').read_text());r=json.loads((output/'ratings.json').read_text())
 assert k['packet_sha256']==r['packet_sha256']==hashlib.sha256((output/'packet.json').read_bytes()).hexdigest()
 expected={(q['n'],o) for q in p['questions'] for o in q['options']}
 assert len(r['ratings'])==len(expected) and {(x['n'],x['option']) for x in r['ratings']}==expected
 splits={q['n']:q['split'] for q in p['questions']};counts={}
 for row in r['ratings']:
  assert all(row.get(a) in ('pass','fail','uncertain') for a in AXES) and row.get('reason')
  arm=k['mapping'][str(row['n'])][row['option']];c=counts.setdefault(splits[row['n']],{}).setdefault(arm,Counter());c['pairs']+=1
  for a in AXES:c[a+'_'+row[a]]+=1
  c['all_pass']+=all(row[a]=='pass' for a in AXES)
 result=dict(counts=counts,packet_sha256=k['packet_sha256'],ratings_sha256=hashlib.sha256((output/'ratings.json').read_bytes()).hexdigest(),limitation='One seed, reused50 validation +24 development diagnostics; subjective assistant ratings, not independent human accuracy.10 reviewed additions and590 blanket-authorized additions are not600 per-item human quality labels.')
 (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','score']);p.add_argument('--directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.action=='prepare':prepare(a.directory,a.output)
 else:score(a.output)
