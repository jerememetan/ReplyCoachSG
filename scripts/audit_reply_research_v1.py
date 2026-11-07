"""Read-only target/output audit; no product test access or training."""
import hashlib,json,re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def norm(text):return ' '.join(re.findall(r"[a-z0-9']+",text.casefold()))
def starts_can(text):return bool(re.match(r'^can\b',text.strip(),re.I))
def rows(path):
 d=json.loads(path.read_text()) if path.suffix=='.json' else [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
 return d['items'] if isinstance(d,dict) else d

def stats(records):
 pairs=[r['replies'] for r in records];local=[p['local'] for p in pairs]
 starts=Counter(' '.join(norm(t).split()[:2]) for t in local)
 values=Counter(norm(t) for t in local)
 return dict(items=len(pairs),can_initial=sum(starts_can(t) for t in local),can_interjection_initial=sum(bool(re.match(r'^\s*can\s*[,!]',t,re.I)) for t in local),exact_normalized_copy=sum(norm(p['standard'])==norm(p['local']) for p in pairs),unique_normalized_local=len(values),most_common_openings=starts.most_common(8),repeated_complete_local=[dict(reply=t,count=n) for t,n in values.most_common(6) if n>1])

def main():
 sources={'original190':'data/dev/train.jsonl','curated190':'data/distillation/shared_intent_v1_reviewed.json','expanded790':'data/training_expansion/blanket600_v1/expanded.jsonl'}
 audit=dict(targets={},sources_sha256={})
 for name,f in sources.items():
  p=ROOT/f;rs=rows(p);audit['targets'][name]=stats(rs);audit['sources_sha256'][f]=hashlib.sha256(p.read_bytes()).hexdigest()
 orig={r['id']:r for r in rows(ROOT/sources['original190'])};curated={r['item_id']:r for r in rows(ROOT/sources['curated190'])}
 changes=[]
 for id,old in orig.items():
  new=curated[id]
  if starts_can(new['replies']['local']) and not starts_can(old['replies']['local']):
   changes.append(dict(item_id=id,original=old['replies']['local'],curated=new['replies']['local']))
 audit['introduced_can_openings']=changes
 audit['curated_vs_original_changes']=dict(local_normalized=sum(norm(r['replies']['local'])!=norm(orig[id]['replies']['local']) for id,r in curated.items()),standard_normalized=sum(norm(r['replies']['standard'])!=norm(orig[id]['replies']['standard']) for id,r in curated.items()))
 audit['curation']=json.loads((ROOT/sources['curated190']).read_text())['audit']
 for file in ('artifacts/standard_val_v1/full_lr3e4.json','artifacts/standard_val_v1/full_lr1e4.json','artifacts/standard_val_v1/lora_lr3e4.json'):
  r=json.loads((ROOT/file).read_text());audit.setdefault('overfitting',{})[r['variant']]=dict(old_endpoint=r['old_endpoint'],best_epoch=r['best_epoch'],best_validation_loss=r['best_validation_loss'])
 audit['human_pair_review']=json.loads((ROOT/'experiments/hybrid_pair_ab_v1_results.json').read_text())
 actual=json.loads((ROOT/'artifacts/hybrid_pair_ab_v1/paired_outputs.json').read_text())['outputs']
 audit['fresh_outputs']={arm:dict(cases=len(actual),can_initial=sum(starts_can(r[arm]['local']) for r in actual),exact_normalized_copy=sum(norm(r[arm]['standard'])==norm(r[arm]['local']) for r in actual)) for arm in ('hybrid','earlier_paired')}
 # State overlap is flagged, never assumed semantic leakage from message-only matches.
 validation=rows(ROOT/'data/dev/validation.jsonl')
 def state(r):
  ctx=r.get('context',[t['text'] for t in r.get('session_memory',[])])
  msg=r['message'];msg=msg.get('text') if isinstance(msg,dict) else msg
  return norm(' | '.join(ctx)+' | '+msg)
 trainrows=rows(ROOT/sources['expanded790'])
 audit['expanded_validation_exact_full_text_overlaps']=len({state(r) for r in trainrows}&{state(r) for r in validation})
 # Original role annotations exist; app's plain context API still guesses roles.
 audit['nonalternating_context_items']={}
 for name,rs in [('original_train',list(orig.values())),('validation',validation)]:
  count=0
  for r in rs:
   turns=r['session_memory'];expected=['user' if (len(turns)-i)%2 else 'peer' for i in range(len(turns))]
   count += [t['speaker'] for t in turns]!=expected
  audit['nonalternating_context_items'][name]=count
 output=ROOT/'experiments/reply_research_v1_audit.json';output.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in audit.items() if k not in ('introduced_can_openings','sources_sha256')},indent=2))
 print('New Can openings introduced by curation:',len(changes))

if __name__=='__main__':main()
