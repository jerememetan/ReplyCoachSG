"""Freeze fresh diagnostic chats; assistant references stay outside model inputs."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# No existing final test or original validation records are read.
CASES=[
('time','I finish my shift at 7:15 tonight.','Then we can have dinner afterwards.','Shall we meet for dinner at 6:45?','I finish at 7:15. Could we meet after that?','Do not agree to 6:45; shift ends 7:15.'),
('time','I need to leave campus by 4:30.','Okay, we can keep the meeting short.','Can we start our one-hour meeting at four?','Could we start earlier? I need to leave by 4:30.','A one-hour meeting at four exceeds leaving time.'),
('availability','I cannot rehearse on Wednesday, but Thursday is free.','I will check the others.','Everyone picked Wednesday evening. Can you come?','I cannot do Wednesday. Is Thursday still possible?','Wednesday unavailable; do not accept it.'),
('availability','I can attend the workshop in the morning only.','Let me check the sessions.','Would you prefer the morning or afternoon session?','The morning session, please.','Choose morning; afternoon unavailable.'),
('ownership','I still have the stapler you lent me.','I need it for my poster.','Could you return my stapler before lunch?','Sure, I will return your stapler before lunch.','USER borrowed PEER stapler; USER returns it.'),
('ownership','You can keep my umbrella until the rain stops.','Thanks, I have it in my bag.','Is it okay if I return your umbrella after class?','Yes, after class is fine.','PEER holds USER umbrella; grant permission, do not offer to return it.'),
('direction','I missed the instructions for the survey.','I took notes during the briefing.','Would you like me to send you my notes?','Yes please, that would help.','PEER offers notes to USER; do not reverse sender.'),
('direction','I saved the tutorial recording on my laptop.','I missed that tutorial.','Could you send me the recording tonight?','Sure, I will send it tonight.','USER sends recording to PEER.'),
('completed','I uploaded our slides a few minutes ago.','Thanks for handling that.','Have you uploaded them yet?','Yes, I have already uploaded them.','Already uploaded; do not promise first upload later.'),
('unfinished','I have not started editing the video.','We still have a little time.','Is the edited video ready?','Not yet, I have not started editing it.','Not started; do not claim completed.'),
('question','I am waiting outside seminar room 5B.','I am coming upstairs now.','Which room are you outside?','I am outside seminar room 5B.','Answer known room 5B.'),
('question','I left my green water bottle in the studio.','There are a few bottles here.','What colour is yours?','Mine is green.','Answer colour, not lost location.'),
('uncertainty','I have not checked whether the lab is open tonight.','We might need somewhere to work.','Does it close at nine?','I am not sure. Let me check the opening hours.','Opening time explicitly unknown; do not assert nine.'),
('uncertainty','I do not know how to use the new scanner.','Neither do I.','Can you show me how to scan this?','I am not sure how either. Shall we ask someone?','USER lacks skill; do not confidently offer instruction.'),
('constraint','My lunch budget is eight dollars today.','We can find somewhere nearby.','Want to try the eighteen-dollar lunch set?','That is over my budget. Shall we find something cheaper?','Do not accept eighteen dollars as within eight-dollar budget.'),
('constraint','My wrist is still sore, so I cannot carry heavy things.','Hope it gets better soon.','Can you help me carry these heavy boxes?','I cannot carry them with my wrist like this. Could we ask someone else?','Do not agree to heavy lifting despite stated inability.'),
('support','The presentation is finally over.','You practised for it all week.','I am so relieved I did not freeze halfway through.','Glad it went okay. You can finally relax now.','Acknowledge relief; no unrelated meeting suggestion.'),
('support','I have been doing most of the project on my own.','Have the others been helping?','Not really, and I am feeling quite overwhelmed.','That sounds exhausting. Want to talk through what is left?','Respond supportively to overwhelm.'),
('support','My interview results came out this morning.','How did it go?','I did not get the internship.','Sorry to hear that. Do you want to talk about it?','Acknowledge disappointment; do not congratulate success.'),
('health-chat','I have a clinic appointment after class.','What happened?','There is a painful bump on my eyelid, so I am getting it checked.','That sounds uncomfortable. Hope the appointment goes okay.','Supportive peer reply, no invented own appointment or diagnosis.'),
('student-chat','Want anything from the toast stall?','I have not had breakfast yet.','Kaya toast or peanut toast?','Kaya toast, please.','Choose offered food or valid refusal; no unrelated food demand.'),
('student-chat','I am getting a drink before the lecture.','I could use one too.','What would you like me to get you?','An iced tea, please.','Answer drink request; plausible preference allowed.'),
('student-chat','I found the missing citation in our report.','Nice, that was the last thing we needed.','I have fixed it and sent you the updated file.','Thanks, I will check the updated file.','Acknowledge completed correction; do not ask peer to fix it again.'),
('clarification','I can meet after my tutorial.','We could work on the poster together.','Meet me there when you finish.','Sure, where should I meet you?','Location unspecified; clarification is appropriate.')]
def main():
 out=ROOT/'data/evaluation/component_baseline_v1';out.mkdir(parents=True,exist_ok=True)
 inputs=[];refs=[]
 for n,(category,user,peer,message,standard,expectation) in enumerate(CASES,1):
  # Explicit roles avoid the app's alternating-role inference.
  key=f'COMP1-{n:03d}';inputs.append(dict(item_id=key,context=[user,peer],speakers=['user','peer'],message=message,split='diagnostic',group_id=key))
  refs.append(dict(item_id=key,category=category,standard=standard,expectation=expectation,label_source='Assistant authored; alternative valid drafts allowed, not human gold'))
 (out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n');(out/'reference_anchors.json').write_text(json.dumps(refs,indent=2)+'\n')
 original=json.loads((ROOT/'data/public/native_rewrite_v2/original_repaired.json').read_text());ids={'ITEM-A-013','ITEM-B-006','ITEM-C-027','ITEM-C-044'}
 examples=[{k:r[k] for k in ('item_id','standard','local')} for r in original if r['item_id'] in ids];assert len(examples)==4
 (out/'style_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
 manifest=dict(cases=24,training=False,purpose='Model/component capability diagnostic; no final benchmark claim',limitations=['Assistant authored and correlated familiar scenario families, not representative human evaluation.','No original final test read. These cases become development material after inspection.','Model families, sizes, pretraining, quantization and prompt interfaces differ; not a causal size ablation.','Explicit-plan route uses extra calls/tokens; not matched compute.','Correct-reference-anchor rewriting isolates style capability; actual-anchor rewriting measures pipeline behavior.'])
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Frozen24 contexts,24 English reference anchors,4 training-only style examples')
if __name__=='__main__':main()
