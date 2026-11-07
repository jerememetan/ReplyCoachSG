"""Freeze a small, explicitly audited capacity/transfer diagnostic (not product gold)."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/training_expansion/rewrite_diagnostic_v3'
TRAIN={'A':[2,5,13,18,21,27,36,41], 'B':[6,16,21,38,46,53], 'C':[5,7,20,27,31,39,44,59,60,3]}
VAL={'D':[5,18,19,22,28,35,52,55], 'E':[29,34,35,56]}
# English-only fidelity corrections: LOCAL remains byte-for-byte original.
FIX={'ITEM-A-013':'What would you like? I will buy it for you.',
'ITEM-B-016':'Yes, the slides are on the course page already.',
'ITEM-C-020':'Not yet. The lecturer is not here either.',
'ITEM-C-027':"It is okay. Rest first; we can eat together another time.",
'ITEM-C-044':'The lecturer is inside already, but class has not started yet.',
'ITEM-D-018':'Sorry, I have something on tomorrow. What about the day after tomorrow?',
'ITEM-D-022':"I am going to finish soon. What is your answer to the first question?",
'ITEM-E-034':'Yes, I have already booked it.',
'ITEM-E-035':'Why did you wait until the last minute to tell me?'}
def main():
 rows=json.loads((ROOT/'data/public/native_rewrite_v2/original_repaired.json').read_text()); lookup={r['item_id']:r for r in rows};OUT.mkdir(parents=True,exist_ok=True)
 partitions={}
 for split,groups in [('train',TRAIN),('validation',VAL)]:
  selected=[]
  for author,nums in groups.items():
   for n in nums:
    key=f'ITEM-{author}-{n:03d}';r=dict(lookup[key]);r.update(split=split,standard=FIX.get(key,r['standard']),standard_variants=[FIX.get(key,r['standard'])],audit='Assistant inspected for standalone anchor-to-LOCAL fidelity; not human gold. Original LOCAL verbatim.')
    assert r['local']==lookup[key]['local'];selected.append(r)
  partitions[split]=selected;(OUT/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
 assert not ({r['group_id'] for r in partitions['train']}&{r['group_id'] for r in partitions['validation']})
 norm=lambda s:re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()
 assert not ({norm(r['local']) for r in partitions['train']}&{norm(r['local']) for r in partitions['validation']})
 # These English meaning controls have no prescribed Singlish answer; inspect meaning/style separately.
 probes=[('availability-positive','Friday afternoon works for me.'),('availability-negative','Friday afternoon does not work for me.'),('booking-future','I will book the room later.'),('booking-complete','I have already booked the room.'),('owner-you','You can go first.'),('owner-me','I can go first.'),('not-yet','I have not finished the slides yet.'),('already','I have already finished the slides.'),('reassure','No worries, just let me know when you are done.'),('support','That sounds exhausting. Get some rest first.'),('uncertainty','I am not sure; let me check first.'),('return','I will return your charger after my phone finishes charging.')]
 (OUT/'meaning_probes.json').write_text(json.dumps([dict(item_id='DIAG3-'+k,standard=s,context=[],speakers=[],message='',group_id='diagnostic-probes',split='probe') for k,s in probes],indent=2)+'\n')
 (OUT/'manifest.json').write_text(json.dumps(dict(train=24,validation=12,probes=12,english_corrections=FIX,source_sha256=hashlib.sha256((ROOT/'data/public/native_rewrite_v2/original_repaired.json').read_bytes()).hexdigest(),limitations=['Capacity probe intentionally repeats training examples; training fit is not generalization.','Authors A/B/C versus D/E separate gradient updates here; original data may have appeared in previous project checkpoints or public pretraining.','Existing original scenarios overlap; this is author-disjoint transfer, not topic-disjoint student generalization.','English corrections and controls assistant authored, not human labels.','Meaning probes are diagnostic development inputs; no reference answers and no automatic quality certification.']),indent=2)+'\n')
 review=['# Small rewrite diagnostic review','', 'LOCAL targets are original; English anchors assistant audited. Not human gold.','']
 for split,items in partitions.items():
  review.extend([f'## {split}',''])
  for r in items:review.extend([f"**{r['item_id']}**",f"- English: {r['standard']}",f"- Original Singlish: {r['local']}",''])
 (OUT/'review.md').write_text('\n'.join(review))
 print({k:len(v) for k,v in partitions.items()})
if __name__=='__main__':main()
