"""Freeze new assistant-authored reward diagnostics; not human gold or training data."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
# group, history, incoming, two acceptable drafts, two unacceptable drafts.
CASES=[
('constraint', [('user','My seminar ends at ten thirty. I cannot leave before then.')], 'Can you meet me at ten?', ["I can meet at ten forty instead.","Ten cannot, my seminar ends ten thirty."], ["Ten works, see you then.","Can, I'll leave at ten to meet you."]),
('constraint', [('user','I am allergic to peanuts. Please avoid peanuts in my food.')], 'Would you like peanut sauce on yours?', ["No peanut sauce for mine, please.","No thanks, I cannot eat peanuts."], ["Yes, add peanut sauce to mine.","Can, put extra peanuts for me."]),
('constraint', [('user','Please print exactly six copies, all in black and white.')], 'Confirm the number and colour for me?', ["Six copies in black and white, please.","Six copies, black and white only."], ["Six colour copies, please.","Eight copies, black and white only."]),
('constraint', [('peer','The equipment desk closes at four.'),('user','I can only reach campus at five today.')], 'Should I ask them to keep your microphone there for collection?', ["Could you collect it before four and keep it for me?", "I reach at five, can you collect before four for me?"], ["Yes, I'll collect it from the desk at five.","Can, I reach desk at five before they close."]),
('constraint', [('peer','Use the west entrance; the east entrance is closed.'),('user','Got it, west entrance.')], 'Which entrance are you heading to?', ["I'm heading to the west entrance.","West entrance, see you there."], ["I'll use the east entrance since it is open.","East entrance, still open what."]),
('role', [('peer','I left my tablet at your place.'),('user','I have your tablet here.')], 'Can you bring my tablet to my studio?', ["I'll bring your tablet to your studio.","Can, I'll bring your tablet to your studio."], ["Please bring my tablet back to my studio.","You return my tablet to my studio, can?" ]),
('role', [('user','I need someone to cover my booth shift.'),('peer','I can cover it for you.')], 'What time does your shift start?', ["My shift starts at eleven.","My shift start eleven, thanks for covering."], ["Thanks, I will cover your shift at eleven.","I'll cover your shift, start eleven right?" ]),
('role', [('peer','I have your jacket. I borrowed it yesterday.'),('user','Thanks for keeping it safe.')], 'Can I return your jacket after lunch?', ["After lunch is fine, thanks for returning it.","After lunch can, thanks for returning my jacket."], ["I'll return your jacket after lunch.","Can, I'll return your jacket later." ]),
('role', [('user','Could you review my abstract?'),('peer','Sure, send your abstract to me.')], 'When can you send me your abstract?', ["I'll send my abstract this evening.","I'll send my abstract tonight."], ["Please send your abstract so I can review it.","You send your abstract, I'll review tonight." ]),
('role', [('user','I am at the station waiting to pick you up.'),('peer','I am still on the train.')], 'I will arrive in twelve minutes. Can you wait?', ["I'll wait at the station for you.","Can, I'll wait at station for you."], ["I'll get off the train in twelve minutes.","I'm still on train, reaching in twelve minutes." ]),
('choice', [], 'Could you help me pack the exhibition materials tonight?', ["I'm busy tonight, but I can help tomorrow.","Tonight cannot, can help tomorrow instead."], ["The exhibition materials are exhibition materials.","Pack what pack, exhibition material material." ]),
('choice', [], 'Shall we eat noodles together after the workshop?', ["Yes, let's get noodles after the workshop.","I prefer rice, can we get that instead?"], ["I have converted the noodles into a workshop.","Workshop already ate the noodles person." ]),
('choice', [('peer','There are two booths, one near the lifts and one outside the hall.')], 'Could you leave the flyers at the booth?', ["Which booth should I leave them at?", "Lift side or outside hall booth?"], ["I'll leave the booth inside the flyers.","Put booth inside flyer can already." ]),
('choice', [], 'Would you like to join the rehearsal on Monday?', ["I can't join on Monday, but thanks for inviting me.","Monday can, I'll join you all."], ["Monday is the rehearsal of would like.","Join Monday the rehearsal rehearsal who." ]),
('choice', [('user','I do not know which part of the draft you mean.')], 'Could you shorten that section?', ["Which section would you like me to shorten?", "You mean which section?"], ["I'll shorten the section I definitely know you mean.","I know exactly which one already, no need tell me." ]),
('new_detail', [], 'Have you finished preparing your tutorial slides?', ["I finished them this afternoon.","Not yet, still got two slides to finish."], ["Your question has five slides of afternoon.","Finished the finish of preparing preparation." ]),
('new_detail', [], 'Which room did the committee assign for the briefing?', ["The committee assigned room C eighteen.","Room C eighteen, committee confirm already."], ["The room assigned the committee a briefing person.","Room is committee briefing assigned person." ]),
('new_detail', [], 'Did the lecturer approve our revised project topic?', ["Yes, she approved it this morning.","Not yet, she said she'll reply tomorrow."], ["The project topic is the lecturer's approval of a person room.","Lecturer topic approval project topic project." ]),
('new_detail', [], 'When is the final deadline for the club application?', ["It's next Wednesday at noon.","Next Wednesday noon, don't miss it."], ["The deadline is the colour of application.","Deadline is club application Wednesday person colour." ]),
('new_detail', [], 'How many teammates are joining our practice session?', ["Four teammates have confirmed.","Four confirm coming already."], ["The number of teammates is a practice session verb.","Teammates number is session practice verb." ]),
]

def main():
    out=ROOT/'data/reward_diagnostics/challenge_v1.jsonl'
    if out.exists():raise FileExistsError('Never overwrite frozen diagnostic labels')
    rows=[]
    for i,(group,history,message,good,bad) in enumerate(CASES):
        for j,draft in enumerate(good+bad):
            rows.append(dict(id=f'REWARD-v1-{i:02}-{j}',scenario_id=f'REWARD-v1-{i:02}',group=group,
                conversation=history,message=message,style='STANDARD' if j%2==0 else 'LOCAL',draft=draft,
                acceptable=j<2,label_source='Codex-authored diagnostic expectation; NOT human gold',
                rationale='Respects roles and explicit facts; valid refusal, clarification and plausible unknown details allowed.' if j<2 else
                    ('Violates an explicit constraint.' if group=='constraint' else 'Reverses the established actors/ownership.' if group=='role' else 'Incoherent or explicitly contradicts stated user uncertainty.')))
    protected=[]
    for split in ('train','validation'):
        protected += [json.loads(l)['message']['text'].strip().casefold() for l in (ROOT/f'data/dev/{split}.jsonl').read_text().splitlines()]
    assert not any(c['message'].strip().casefold() in protected for c in rows)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in rows))
    print('Frozen',len(rows),'cases in',len(CASES),'scenario clusters. SHA256',hashlib.sha256(out.read_bytes()).hexdigest())

if __name__=='__main__':main()
