"""Freeze new development-only scenarios before checkpoint-duration experiments."""
import json,re
from pathlib import Path
OUT=Path('data/reward_diagnostics/duration_probe_v1.jsonl')
CASES=[
([('peer','The robotics workshop was much harder than I expected.'),('user','Same, that last exercise took me ages.')],'At least our robot finally moved!','Acknowledge shared success or difficulty coherently.'),
([('peer','I sent the revised survey to the team.'),('user','Thanks for handling that.')],'No problem, happy to help.','A coherent closing acknowledgement is enough.'),
([('user','I found a quiet table near the windows.')],'Nice, I will come find you after my tutorial.','Acknowledge peer joining after tutorial, without reversing who found the table.'),
([('peer','The rehearsal went well today.')],'Thanks for helping me practise the lines.','Respond coherently to thanks; no exact reference wording required.'),
([('user','I brought my bike lock to campus.')],'Can I use your lock for my bicycle until this evening?','User may lend/refuse own lock; do not promise to borrow it from peer.'),
([('user','I do not have an HDMI cable with me.')],'I have an extra cable. Would you like to use it?','Accept/refuse peer offer coherently; plausible alternatives allowed.'),
([('peer','I saved a copy of the attendance list.'),('user','I lost mine when my laptop restarted.')],'Do you want me to forward my copy to you?','User should receive offered copy or coherently decline; do not reverse sending direction.'),
([('user','Your parcel arrived at my hostel reception and I collected it.')],'Could you pass my parcel to me before dinner?','User gives peer the peer parcel or proposes coherent alternative.'),
([('peer','You lent me your drawing tablet last week.')],'Can I give your tablet back at our next meetup?','User accepts receiving own tablet or gives coherent return arrangement.'),
([('user','I cannot leave the examination room until 3 pm.')],'Can you meet me outside at 2 pm?','Respect explicit inability; valid later counterproposal/refusal allowed.'),
([('peer','The equipment counter shuts at 4 pm.'),('user','My seminar runs from noon until 5 pm.')],'Can you pick up the lens after your seminar today?','Do not promise pickup after the stated closing time without a coherent alternate arrangement.'),
([('user','My doctor told me not to run on my injured knee this week.')],'Want to do a long run with me tomorrow morning?','Respect no-running constraint; refusal or another suitable activity allowed.'),
([('user','I cannot eat dairy products.')],'Should I buy the cheese pasta for you?','Do not accept the offered dairy meal; a valid alternative is allowed.'),
([('user','I have drafted the questions but have not sent the form yet.')],'Have you sent the form to everyone?','Answer sent-status coherently; preserve status across styles. Progress timing may be uncertain.'),
([('user','I returned your library book at the front desk yesterday.')],'Do you still have my book with you?','Do not deny the stated return or claim the book is still with user.'),
([('peer','Our bus departs from Bay 14 at the south terminal.')],'Which bay do we need to queue at?','Use the explicit Bay14 rather than a different or uninformative bay.'),
([('user','I am standing beside the fountain, not the ticket booth.')],'Where should I look for you?','Use the stated fountain location; do not copy the excluded ticket booth.'),
([('peer','Can you join the baking session on Sunday?'),('user','Sunday afternoon works for me.')],'Great, it starts at 2 in the teaching kitchen.','Acknowledge the plan; changing availability is allowed as an explicit refusal/counterproposal.'),
([('user','Do you want to come to the board-game cafe with us?')],'I cannot come this time. I have a family dinner.','Acknowledge refusal; do not confirm attendance at the same cafe outing.'),
([('user','Can you review my scholarship statement this week?')],'Sure, send it over when you are ready.','User sends own statement or gives coherent timing; do not ask peer to send their statement instead.'),
([('peer','I forgot the badminton shuttlecocks.'),('user','I have a fresh tube in my bag.')],'Could you bring your shuttlecocks to the court?','User offers/refuses bringing own shuttlecocks, without reversing who owns them.'),
([('peer','I have never used this audio mixer and I do not know its controls.')],'I tried turning it on but still cannot get any sound.','Do not ask the explicitly uninformed peer to teach the controls; offering own help or seeking other help is valid.'),
([('user','I have never borrowed equipment from the media desk.')],'I borrow cameras there every week and know the procedure.','Can ask experienced peer for help or give another coherent response.'),
([('peer','I knocked over some water on the table, sorry.')],'Your notes look dry, thankfully.','Respond coherently to apology/near miss without inventing explicit damage as already stated fact.'),
]
def main():
 if OUT.exists():raise FileExistsError(OUT)
 paths=[Path('data/dev/train.jsonl'),Path('data/dev/validation.jsonl'),Path('data/reward_diagnostics/standard_focus_v1.jsonl'),Path('data/synthetic/standard_focus_v1/targeted.jsonl')]
 normalize=lambda text:re.sub(r'\W+',' ',text.casefold()).strip()
 protected={normalize(json.loads(l)['message']['text']) for p in paths for l in p.read_text().splitlines()}
 assert len(CASES)==24 and len({normalize(c[1]) for c in CASES})==24
 assert not protected & {normalize(c[1]) for c in CASES}
 rows=[dict(id=f'DURATION-DEV-{i:03}',split='dev',session_memory=[dict(speaker=s,text=t) for s,t in context],message=dict(text=message),replies=dict(standard='Evaluation only.',local='Evaluation only lah.'),adequacy_guidance=guidance,provenance='New assistant-authored development probe; not human gold; never training.') for i,(context,message,guidance) in enumerate(CASES,1)]
 OUT.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));print('Frozen24 fresh development scenarios; no normalized protected-message overlap.')
if __name__=='__main__':main()
