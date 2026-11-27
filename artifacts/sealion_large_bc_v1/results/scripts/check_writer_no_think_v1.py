import importlib.util,json,time
from pathlib import Path
ROOT=Path('/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_sealion_large_bc_v1')
spec=importlib.util.spec_from_file_location('writer',ROOT/'scripts/sealion_large_bc_data.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
plan=json.loads((ROOT/'data/authoring_plan.json').read_text());accepted=m.loadlines(ROOT/'authoring/accepted.jsonl');seen={r['slot'] for r in accepted}
pending=[s for s in plan['slots'] if s['split']=='train' and s['slot'] not in seen]
kinds=['acknowledge','confirm','invite','ownership','negation','timing'];chosen=[]
for kind in kinds:
 case=next(s for s in pending if s['kind']==kind and s['slot'] not in {x['slot'] for x in chosen});chosen.append(case)
bank=json.loads((ROOT/'data/author_style_fewshots.json').read_text())['examples'];output=ROOT/'authoring/writer_no_think_diagnostic_v1.json'
assert not output.exists()
report=dict(purpose='Writer completion/speed and assistant sample audit; diagnostic outputs are NOT training data.',think=False,writer=m.WRITER,cases=[],student_training_unchanged=True)
for i,case in enumerate(chosen):
 examples=[bank[0],bank[1],bank[2+i%(len(bank)-2)]]
 demo='\n\n'.join(m.render(r)+'\nEnglish draft: '+r['standard']+'\nNatural Singlish: '+r['local'] for r in examples)
 prompt=m.WRITER_RULES+'\nSTYLE DEMONSTRATIONS (follow the style; invent unrelated situations, do not clone these chats):\n'+demo+'\nUse JSON history speakers Friend and You. incoming_friend MUST be a NEW message by Friend. you_drafts MUST contain TWO replies by You TO that incoming_friend. Never confuse incoming_friend with a candidate reply. Follow requested speaker patterns (user=You, peer=Friend).\nRequested specifications in order:\n'+json.dumps([case])+'\nRecent accepted examples to avoid repeating:\n'+json.dumps([m.text(r) for r in accepted[-15:]],ensure_ascii=False)
 schema={'type':'object','properties':{'items':{'type':'array','items':m.ITEM,'minItems':1,'maxItems':1}},'required':['items'],'additionalProperties':False}
 start=time.monotonic()
 response=m.request('/api/chat',dict(model=m.WRITER,stream=False,think=False,format=schema,keep_alive='30m',messages=[dict(role='user',content=prompt)],options=dict(temperature=.7,seed=904250+i,num_ctx=16384,num_predict=6000,num_thread=4)))
 record=dict(specification=case,seconds=time.monotonic()-start,done_reason=response.get('done_reason'),eval_count=response.get('eval_count'))
 try:
  assert response.get('done') and response.get('done_reason')!='length'
  raw=json.loads(response['message']['content']);row=m.structural(raw['items'][0]);assert row['speakers']==case['history_pattern'];record.update(structural_pass=True,candidate=row)
 except Exception as e:record.update(structural_pass=False,error=repr(e),raw=response.get('message',{}).get('content'))
 report['cases'].append(record);m.save(output,report);print(json.dumps(record,ensure_ascii=False),flush=True)
print('WRITER_NO_THINK_DIAGNOSTIC_COMPLETE',flush=True)
