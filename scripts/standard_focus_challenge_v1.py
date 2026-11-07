"""Freeze fresh assistant-authored diagnostic contexts before training; never training labels."""
import json
from pathlib import Path
# Each case records an adequacy requirement, not a mandatory exact answer.
CASES=[
([('user','I have the keys to the music room.')],'May I take your keys until rehearsal ends?','User may lend own keys; do not borrow from peer.'),
([('user','I left my access card at home.')],'You can use my spare card if you need it.','Accept/decline peer offer; do not offer own missing card.'),
([('peer','My laptop broke.'),('user','I saved a backup of the slides.')],'Can you email me the backup?','User sends backup to peer or gives coherent refusal.'),
([('user','I missed the announcement about the field trip.')],'I recorded it. Shall I send you the audio?','Accept/decline receiving peer recording; do not promise to send own recording.'),
([('peer','I cannot figure out how to open the equipment locker.')],'I tried the handle again but I still cannot open it.','Do not ask peer for instructions as if they already know; suggest help or another method.'),
([('user','I have never connected a projector before.')],'I set these projectors up every week.','Can ask peer to help connect projector.'),
([('user','Are you coming for the overnight camp?')],'Sorry, I am sick and will stay home.','Acknowledge refusal; do not confirm peer attendance.'),
([('user','Are you joining our volunteer shift?')],'Yes, I have signed up for the same shift.','Acknowledge attendance; do not interpret as refusal.'),
([('user','My lab ends at 5 pm and I cannot leave early.')],'Could we meet at 4 pm today?','Decline/propose later than lab end; do not agree to four.'),
([('user','I must catch a flight at noon and leave campus by 9 am.')],'Can we meet on campus at 11 am?','Reject incompatible time or propose compatible alternative.'),
([('user','I have only written the introduction to my essay.')],'Have you finished the whole essay already?','Do not claim whole essay complete.'),
([('user','I submitted the final application this morning.')],'Have you sent in the application yet?','Acknowledge submission rather than promise to start it.'),
([('peer','The interview is in Tower C, room 610.')],'Which room should we go to for the interview?','Answer Tower C room610 or compatible acknowledgement; do not give different room.'),
([('peer','The coach changed our practice location but has not told us the new venue.')],'Do you know where practice is now?','Plausible selectable location allowed by policy; checking/uncertainty also valid; do not assume contradiction from unknown alone.'),
([('user','I will attend the safety briefing and you will be away.')],'Could you tell me what I miss afterwards?','User should update peer; do not ask peer what peer missed.'),
([('user','I will be away during the election briefing.'),('peer','I can attend it for us.')],'Should I take notes for you?','Accept/decline peer taking notes for user.'),
([('peer','I borrowed your lab coat yesterday.')],'Can I return the coat to you this Friday?','User receives own coat back; do not promise to return peer coat.'),
([('user','I still have the toolkit you lent me.')],'Can you bring my toolkit back on Monday?','User returns peer toolkit or proposes alternative; do not ask to receive it.'),
([('user','I am allergic to eggs.')],'Want me to get you the egg sandwich?','Refuse egg sandwich or ask egg-free alternative.'),
([('user','I would like a vegetarian dinner.')],'Should I order the mushroom rice for you?','Coherent acceptance/refusal; do not invent requirement for meat.'),
([('peer','Could you help me carry these boxes?'),('user','I hurt my wrist and cannot lift anything.')],'Can you lift the heaviest box for me then?','Respect inability to lift; offer different help or decline.'),
([('peer','The shop closes at 6 pm.'),('user','I finish work at 7 pm.')],'Can you collect the order after work today?','Cannot collect after7 from shop closed6; propose other arrangement.'),
([('user','I have read chapter one but not chapter two.')],'Can you explain chapter two to me?','Valid refusal, offer to read together, or plausible knowledge permitted; do not claim already read chapter2 contrary to context.'),
([('peer','I am outside the west entrance.'),('user','I am at the east entrance.')],'Which entrance are you waiting at?','User is at east; do not copy peer west location.'),
]
if __name__=='__main__':
    path=Path('data/reward_diagnostics/standard_focus_v1.jsonl')
    if path.exists():raise FileExistsError(path)
    path.write_text(''.join(json.dumps(dict(id=f'FOCUS-DIAG-{i:03}',split='dev',session_memory=[dict(speaker=s,text=t) for s,t in ctx],message=dict(text=msg),replies=dict(standard='Evaluation only.',local='Evaluation only lah.'),adequacy_requirement=req,provenance='Assistant-authored frozen diagnostic; not human gold; never train.'),ensure_ascii=False)+'\n' for i,(ctx,msg,req) in enumerate(CASES,1)))
