"""Prepare user-review CSVs; export only explicit GOOD decisions as training candidates."""
import argparse,csv,hashlib,json,random,re
from collections import defaultdict
from pathlib import Path
FIELDS=['id','scenario','conversation','incoming_message','standard_reply','local_reply','decision','corrected_standard','corrected_local','notes']
IMMUTABLE=FIELDS[:6]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def message(item):
 value=item['message'];return value['text'] if isinstance(value,dict) else value

def normalize(text):return re.sub(r'\W+',' ',text.casefold()).strip()
def review_row(item):
 return dict(id=item['id'],scenario=item.get('scenario_group',''),conversation='\n'.join(f"{t['speaker']}: {t['text']}" for t in item['session_memory']),incoming_message=message(item),standard_reply=item['replies']['standard'],local_reply=item['replies']['local'],decision='',corrected_standard='',corrected_local='',notes='')
def write_csv(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as handle:
  writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
def prepare(source,output,batch_size=10,protected=()):
 if output.exists():raise FileExistsError('Fresh output directory required; preserve existing user reviews')
 if batch_size<1:raise ValueError('Batch size must be positive')
 items=[json.loads(l) for l in source.read_text().splitlines() if l.strip()]
 if not items or len({c['id'] for c in items})!=len(items):raise ValueError('Nonempty unique candidate IDs required')
 protected_messages={normalize(message(json.loads(l))) for p in protected for l in p.read_text().splitlines() if l.strip()}
 overlap=[c['id'] for c in items if normalize(message(c)) in protected_messages]
 if overlap:raise ValueError('Protected incoming-message overlap: '+', '.join(overlap))
 groups=defaultdict(list)
 for c in items:
  if not all(isinstance(c['replies'].get(k),str) and c['replies'][k].strip() for k in ('standard','local')):raise ValueError('Both nonempty drafts required')
  groups[c.get('scenario_group','')].append(c)
 rng=random.Random(425)
 for group in groups.values():rng.shuffle(group)
 order=[]
 while any(groups.values()):
  for family in sorted(groups):
   if groups[family]:order.append(groups[family].pop())
 output.mkdir(parents=True);rows=[review_row(c) for c in order]
 write_csv(output/'all_candidates.csv',rows)
 for start in range(0,len(rows),batch_size):write_csv(output/f'batch_{start//batch_size+1:02}.csv',rows[start:start+batch_size])
 (output/'manifest.json').write_text(json.dumps(dict(source=str(source.resolve()),source_sha256=digest(source),count=len(items),human_approved_count=0,status='pending_human_review',order=[c['id'] for c in order],immutable_columns=IMMUTABLE,decision_values=['good','bad','unsure'],batch_size=batch_size),indent=2)+'\n')
 (output/'HOW_TO_REVIEW.txt').write_text('Each row is one fictional conversation and its two proposed replies. Incoming messages are from peer; replies are your next turn.\n\nFill decision with good, bad, or unsure. Leave unseen rows blank.\nGOOD means both replies fit the conversation, preserve approximately the same meaning, and LOCAL sounds naturally Singlish. Plausible selectable details, refusals and counterproposals are allowed; explicit contradictions and reversed roles are not.\nIf a reply needs fixing, use corrected_standard and/or corrected_local, then mark good only when you approve the corrected pair. Bad and unsure remain excluded. Notes are optional.\nKeep the first six columns unchanged so your decisions stay tied to the exact examples.\nReview either batch files or all_candidates.csv, not both. Start with batch_01.csv.\nThese are pending AI drafts. No row is considered human approved until you explicitly mark good. Import produces a reviewable candidate file, not a training job.\nGood/bad alone is not a DPO preference pair. DPO additionally needs a preferred and rejected reply to the SAME conversation/style. Both-bad or unsure should not force a winner.\n')
 print(f'Prepared{len(rows)} candidates in{(len(rows)+batch_size-1)//batch_size} batches;0 human approvals.')
def import_reviews(manifest_path,files,reviewer,output):
 if output.exists():raise FileExistsError('Fresh output directory required')
 if not reviewer.strip():raise ValueError('Explicit reviewer identity required')
 manifest=json.loads(manifest_path.read_text());source=Path(manifest['source'])
 if digest(source)!=manifest['source_sha256']:raise ValueError('Candidate source changed')
 candidates={c['id']:c for c in map(json.loads,source.read_text().splitlines())};seen=set();approved=[];decisions=[]
 for path in files:
  with path.open(encoding='utf-8-sig',newline='') as handle:
   reader=csv.DictReader(handle)
   if reader.fieldnames!=FIELDS:raise ValueError('Unexpected review columns')
   for row in reader:
    item_id=row['id']
    if item_id not in candidates or item_id in seen:raise ValueError('Unknown or duplicate reviewed item ID')
    seen.add(item_id);item=candidates[item_id];expected=review_row(item)
    if any(row[k]!=expected[k] for k in IMMUTABLE):raise ValueError('Conversation/draft columns changed; use correction columns')
    decision=row['decision'].strip().lower()
    if decision not in ('','good','bad','unsure'):raise ValueError('Decision must be good, bad, unsure or blank')
    if decision:decisions.append(dict(id=item_id,decision=decision,notes=row['notes'],corrected_standard=row['corrected_standard'],corrected_local=row['corrected_local'],reviewer=reviewer,review_file_sha256=digest(path)))
    if decision!='good':continue
    replies={k:row['corrected_'+k].strip() or item['replies'][k] for k in ('standard','local')}
    if not all(replies.values()) or replies['standard'].strip().casefold()==replies['local'].strip().casefold():raise ValueError('Approved pair requires two nonempty distinct styles')
    approved.append(dict(id=item_id,session_memory=item['session_memory'],message=message(item),replies=replies,scenario_group=item.get('scenario_group',''),provenance={**item['provenance'],'human_review_source_sha256':manifest['source_sha256']},review=dict(reviewer=reviewer,meaning_preserved=True,speaker_correct=True,natural_local=True,facts_supported=True,scope='Explicit user GOOD decision under selectable-draft policy; original or corrected paired targets.',decision='good',notes=row['notes'],original_ai_review=item.get('review'),review_file_sha256=digest(path)),human_review=dict(reviewer=reviewer,decision='good',source_manifest_sha256=digest(manifest_path))))
 output.mkdir(parents=True)
 (output/'approved.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in approved))
 (output/'decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2)+'\n')
 (output/'summary.json').write_text(json.dumps(dict(approved=len(approved),bad=sum(d['decision']=='bad' for d in decisions),unsure=sum(d['decision']=='unsure' for d in decisions),pending=manifest['count']-len(decisions),reviewer=reviewer,source_manifest_sha256=digest(manifest_path),training_started=False,dpo_pairs_created=0),indent=2)+'\n')
 print(f'Exported{len(approved)} explicitly approved paired targets. No training started;0 automatic DPO pairs.')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 a=sub.add_parser('prepare');a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--batch-size',type=int,default=10);a.add_argument('--protected',type=Path,nargs='*',default=[])
 a=sub.add_parser('import');a.add_argument('--manifest',type=Path,required=True);a.add_argument('--reviews',type=Path,nargs='+',required=True);a.add_argument('--reviewer',required=True);a.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.action=='prepare':prepare(a.source,a.output,a.batch_size,a.protected)
 else:import_reviews(a.manifest,a.reviews,a.reviewer,a.output)
