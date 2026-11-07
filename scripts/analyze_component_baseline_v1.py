import json, hashlib, statistics
from pathlib import Path
root=Path('artifacts/component_baseline_v1')
names=['qwen4b','gemma12b','intent_qwen4b','mt5']
reports={n:json.loads((root/f'{n}.json').read_text()) for n in names}
inputs=json.loads(Path('data/evaluation/component_baseline_v1/inputs.json').read_text())
ids=[c['item_id'] for c in inputs]
summary={}
for n,d in reports.items():
 assert d['status']=='complete'
 assert [c['item_id'] for c in d['cases']]==ids
 failures=[]; costs={}
 for c in d['cases']:
  for k,s in c.items():
   if isinstance(s,dict) and 'status' in s:
    if s['status']!='complete': failures.append([c['item_id'],k,s['status']])
    rt=s.get('runtime',{})
    bucket=costs.setdefault(k,{'seconds':0,'prompt_tokens':0,'output_tokens':0,'calls':0,'durations':[]})
    bucket['calls']+=1
    for f in ['seconds','prompt_tokens','output_tokens']: bucket[f]+=rt.get(f,0)
    bucket['durations'].append(rt.get('seconds',0))
  for route,anchor_key in [('direct_local','direct_standard'),('planned_local','planned_standard'),('intent_local','intent_standard')]:
   if route in c:
    a=c[anchor_key]; expected=a if isinstance(a,str) else a['value']['reply']
    assert c[route]['standard_anchor']==expected,(n,c['item_id'],route)
 for b in costs.values(): b['median_seconds']=statistics.median(b.pop('durations'))
 summary[n]={'status':d['status'],'cases':len(d['cases']),'failed_stages':failures,'sha256':hashlib.sha256((root/f'{n}.json').read_bytes()).hexdigest(),'digest':d.get('model_metadata',{}).get('digest'),'stage_costs':costs}
hashes={}
for name in ['component_baseline_v1','intent_control_v1','component_mt5_v1_retry']:
 recipe=json.loads(Path(f'experiments/{name}_recipe.json').read_text())
 bad=[p for p,h in recipe['files_sha256'].items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
 hashes[name]={'files':len(recipe['files_sha256']),'local_hash_mismatches':bad}
packet=[{'input':c,'outputs':{n:reports[n]['cases'][i] for n in names}} for i,c in enumerate(inputs)]
(root/'matched_review.json').write_text(json.dumps(packet,indent=2)+'\n')
result={'scope':'24 assistant-authored development diagnostics; exploratory assistant review, not human gold or locked product test','validation':summary,'frozen_local_source_checks':hashes,'provenance_limit':'Local sources match checks are separate from remote snapshot attestation; output JSON does not embed all source hashes.','findings':[
 {'finding':'Gemma is the strongest candidate for the next user comparison, based on inspected contextual replies and more coherent local rewrites; no numeric quality ranking established.','examples':['COMP1-001','COMP1-002','COMP1-003','COMP1-008','COMP1-012','COMP1-015','COMP1-016']},
 {'finding':'Free planning has mixed effects in Qwen: corrects recording and bottle replies but creates an invalid meeting acceptance.','examples':['COMP1-008','COMP1-012','COMP1-002']},
 {'finding':'Categorical Qwen prototype does not establish intent-conditioning gains. Incoming labels and reply acts are often inappropriate: invitation classified as status request with accept act; unedited video classified as invitation.','examples':['COMP1-001','COMP1-010']},
 {'finding':'A factual constraint list can still yield an incorrect realization: Qwen meeting plan records departure and duration but renderer accepts incompatible start.','examples':['COMP1-002']},
 {'finding':'Correct standard replies can be corrupted by rewriting: Qwen turns $8 counterproposal into refusal of $8, reverses drink buyer, and adds unrelated meal suggestion to clinic well wishes.','examples':['COMP1-015','COMP1-022','COMP1-020']},
 {'finding':'mT5 oracle rewrites independently change negation, ownership and concrete details, so its rewrite failure is not only inherited from STANDARD.','examples':['COMP1-003','COMP1-006','COMP1-010','COMP1-012']},
 {'finding':'Gemma planning is not error-free: its presentation plan assigns presenter facts to user while replying to peer relief. Good final reply does not validate the plan.','examples':['COMP1-017']},
 {'finding':'Unspecified meeting location remains unclarified by chat-model final replies; mT5 invents library.','examples':['COMP1-024']},
 {'finding':'Several emotional-dialogue contexts switch first-person speaker across turns (017-020); treat ownership ambiguity separately from clear timing, negation and colour contradictions.','examples':['COMP1-017','COMP1-018','COMP1-019','COMP1-020']}
], 'next_step':'Use Gemma direct STANDARD plus LOCAL as provisional comparison candidate; retain mT5 baseline. Review style against original LOCAL examples with user before selecting defaults or training. If testing trained intent conditioning, annotate reply acts and compare predicted versus oracle acts; prompted classifier failure does not refute architecture.','default_changed':False,'new_training_submitted':False}
Path('experiments/component_baseline_v1_results.json').write_text(json.dumps(result,indent=2)+'\n')
for n,s in summary.items():print(n,s['cases'],'failures',len(s['failed_stages']),{k:round(v['seconds'],2) for k,v in s['stage_costs'].items()})
print('source checks',hashes)
for name in ['component_baseline_v1','intent_control_v1']:
 p=Path(f'experiments/{name}_launch.json'); d=json.loads(p.read_text())
 if name=='component_baseline_v1':
  for j in d['jobs']:
   if j['id']==355489:j.update(last_verified_state='COMPLETED',exit_code='0:0',local_report=str(root/'gemma12b.json'),cases=24)
 else:d.update(last_verified_state='COMPLETED',exit_code='0:0',local_report=str(root/'intent_qwen4b.json'),cases=24)
 p.write_text(json.dumps(d,indent=2)+'\n')
