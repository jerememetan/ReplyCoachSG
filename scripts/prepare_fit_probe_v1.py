"""Freeze24 STANDARD examples and24 wording-held-out counterparts; never touch the locked test."""
import copy,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/training_diagnostics/fit_probe_v1'
# Paraphrases preserve actors, objects, numbers, negation, timing and desired action.
PARAPHRASES={
86:([('peer','Will you be available to practise with us the following day?')], 'The rehearsal time has not been decided so far.'),
35:([('user','I can spend no more than ten dollars on lunch today.')], 'Shall we get the set that costs twenty dollars?'),
74:([('peer','I have not put one of these display stands together before, either.')], 'Is there anyone we could ask to assist us?'),
48:([('user','The badminton nets went into the cage for sports equipment, rather than the office.')], 'Where have you stored those nets?'),
99:([('peer','Some water from my cup spilled onto this table.')], 'Luckily, it does not seem to have made your bag wet.'),
7:([('user','The tray you use for baking is on the shelf in my kitchen.')], 'Would you take my baking tray to the room where we cook?'),
60:([('user','Yesterday I put the photographs into the folder we share.')], 'Are the photographs uploaded now?'),
68:([('user','Would you like us to ride our bicycles to the park together?')], 'I would prefer to go there on foot.'),
20:([('peer','I can share either the presentation slides or the notes I wrote by hand.')], 'Which of those would you prefer me to send?'),
29:([('user','I have time on Tuesday morning and on Thursday afternoon.')], 'Would three in the afternoon on Thursday suit you?'),
1:([('user','My portable fan is with me, inside my tote.')], 'May I borrow your fan while the briefing takes place outside?'),
2:([('user','My ruler is back in the studio, so it is not with me.')], 'There is a spare ruler in my bag. Would you like to borrow it?'),
3:([('peer','I still have your raincoat in my locker.')], 'Should I take your coat to the bus interchange for you tonight?'),
4:([('user','The tote you lent me for the fair is still with me.')], 'Could you return my tote when we have lunch?'),
11:([('user','I have saved the timetable for the debate.')], 'Would you send me that debate timetable?'),
12:([('peer','I have the newest version of our song list.')], 'Would you like me to share the updated list with you before we practise?'),
21:([('user','I must remain at work until my shift finishes at six this evening.')], 'Would half past five by the shop entrance be a good meeting time?'),
24:([('peer','Costume collection ends at three in the afternoon.'),('user','I will not finish rehearsing until four.')], 'Will you be able to pick up our costumes once rehearsal is over?'),
31:([('user','Beef is something I do not eat.')], 'Would you like a serving of the noodles with beef?'),
33:([('user','Because my ankle is sprained, court sports are not possible for me this week.')], 'Are you able to play basketball with our group tomorrow?'),
41:([('peer','Practice will take place in Studio4, on the second level.')], 'Which studio do I need to go to for practice?'),
42:([('user','I am waiting next to the blue vending machine by the northern gate.')], 'Can you tell me exactly where to find you?'),
52:([('user','This morning I sent in the request for the room.')], 'Is the room request submitted already?'),
55:([('user','After our shift, I handed your badge to the people at the event desk.')], 'Is my badge still with you?'),
}
def norm(s):return re.sub(r'\W+',' ',s.casefold()).strip()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 source=ROOT/'data/human_review/expansion_v1/fresh_candidates.jsonl';all_items={x['id']:x for x in map(json.loads,source.read_text().splitlines())}
 approved_path=ROOT/'data/human_review/expansion_v1/approved_batch01/approved.jsonl';approved={x['id']:x for x in map(json.loads,approved_path.read_text().splitlines())}
 assert len(approved)==10
 original=[];paraphrases=[]
 for number,(context,message) in PARAPHRASES.items():
  item_id=f'SYN-HUMAN-EXP1-{number:03}';item=all_items[item_id];target=approved.get(item_id,item)['replies']['standard']
  base=dict(item_id=f'FIT-{number:03}',source_item_id=item_id,context=[t['text'] for t in item['session_memory']],speakers=[t['speaker'] for t in item['session_memory']],message=item['message'],style='STANDARD',response=target,review_scope='explicit user first-batch approval' if item_id in approved else 'assistant checked; blanket training authorization; not individual human gold')
  assert isinstance(target,str) and target.strip()
  original.append(base)
  paraphrases.append({**copy.deepcopy(base),'item_id':f'PARA-{number:03}','paired_train_id':base['item_id'],'context':[t for _,t in context],'speakers':[s for s,_ in context],'message':message,'review_scope':'assistant-authored/checked meaning-preserving paraphrase; not human gold'})
 assert len(original)==len(paraphrases)==24
 assert {x['source_item_id'] for x in original}&set(approved)==set(approved)
 assert len({norm(x['message']) for x in original+paraphrases})==48
 OUT.mkdir(parents=True)
 for name,rows in [('train',original),('paraphrases',paraphrases)]:
  (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 (OUT/'manifest.json').write_text(json.dumps(dict(training_examples=24,individually_user_reviewed=10,assistant_checked_additional=14,paraphrase_examples=24,task='STANDARD only',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,approved_path]},limitations=['Paraphrases retain underlying scenarios and objects; this measures wording transfer, not24 new independent scenarios or general reasoning.','Reference exact match is memorization/target adherence, not the sole contextual-quality criterion.','No LOCAL training, production promotion, reward labels or held-out-test access.']),indent=2)+'\n')
 print('Frozen24 STANDARD training examples (10 user-reviewed +14 assistant-checked) and24 paraphrase counterparts.')
if __name__=='__main__':main()
