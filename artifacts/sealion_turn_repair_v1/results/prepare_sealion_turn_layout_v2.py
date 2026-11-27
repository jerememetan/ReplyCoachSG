"""Freeze explicit-turn serialization and prospective synthetic holdout cases."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/sealion_turn_repair_v1'
OUT=BASE/'layout_v2'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if path.exists() and path.read_text()!=text:raise FileExistsError(path)
    path.write_text(text)


def main():
    tuples=[
      ('purpose', [('user','I need to figure out how to appeal the hostel allocation.'),('peer','Currently helping my parents, cannot answer a call.'),('user','We can talk afterwards.')], 'Timing not confirmed yet, but what were you hoping to discuss?', 'I wanted to ask about appealing the hostel allocation.', 'Could you call me immediately?'),
      ('purpose', [('user','I want to check whether we used the right measurements for the shelves.'),('peer','I am cycling home.'),('user','Okay, call when you are safely home.')], 'Sure, but what is the call regarding?', 'It is about checking the measurements for the shelves.', 'What did you want to call me about?'),
      ('purpose_unspecified', [('user','Hey, can we speak over the phone for a moment?'),('peer','I cannot right now, my class is starting.'),('user','No rush, another time works.')], 'Not certain when yet; can you say what it is regarding?', "I'll send a message explaining what I wanted to discuss first.", 'Can you call me now?'),
      ('purpose_unspecified', [('user','Got a moment to chat by phone?'),('peer','No sorry, working at the counter.'),('user','Okay, message me when you can.')], 'Will do. Was there something you wanted to ask me?', "I'll explain what I wanted to ask in a message first.", 'What did you want to ask me?'),
      ('clarification', [('user','Could you proofread the abstract? The charts are already checked.'),('peer','Okay, I can spare ten minutes.'),('user','Thanks for helping.')], 'Just checking, do you need the abstract or the charts reviewed?', 'Please proofread the abstract; the charts are already checked.', 'Please review the charts.'),
      ('clarification', [('peer','I have the accommodation form and the transport form.'),('user','Could you forward the accommodation form?'),('peer','Okay, opening my files now.')], 'Which form was it you needed?', 'I need the accommodation form.', 'I need the transport form.'),
      ('status', [('user','I have signed the consent sheet but it is still on my desk; I have not handed it in.'),('peer','The organiser is collecting the sheets today.')], 'Has the organiser got your signed sheet?', 'Not yet. I have signed it but have not handed it in.', 'Yes, the organiser has received it.'),
      ('status', [('user','I have finished editing the audio. The subtitles are still unfinished.'),('peer','I need an update for the production team.')], 'Are the subtitles finished too?', 'The subtitles are not finished yet.', 'Yes, the subtitles are finished.'),
      ('return_direction', [('peer','Your sleeping bag is still at my place; I borrowed it for the overnight hike.'),('user','I need it again next weekend.')], 'Which of us should return the sleeping bag?', 'You should return my sleeping bag to me.', 'I should return your sleeping bag to you.'),
      ('speaker_location', [('user','I am inside the bakery, next to the bread shelves.'),('peer','I am across the road at the fruit stall.')], 'Okay, where should I look for you?', 'I am inside the bakery next to the bread shelves.', 'I am at the fruit stall across the road.'),
      ('availability', [('user','I am taking an exam on Monday morning. Tuesday morning is clear.'),('peer','We need to arrange a practice session.')], 'Shall we use Monday morning?', 'I have an exam on Monday morning. Could we use Tuesday morning?', 'Monday morning works for me.'),
      ('time_constraint', [('user','The repair shop will keep me until 6:30 pm, so I cannot leave before then.'),('peer','I will be nearby this evening.')], 'Does meeting at 5 pm work?', 'I cannot meet at 5 pm; I cannot leave before 6:30 pm.', 'Yes, 5 pm works.'),
      ('budget', [('user','I can spend no more than $18 for the ticket.'),('peer','The only ticket left costs $41.')], 'Would you like me to book that ticket for you?', 'That is over my $18 budget. Please do not book it.', 'Yes, book the $41 ticket for me.'),
      ('correction', [('user','I suggested the rooftop, but it is closed for maintenance.'),('peer','Do you have another place in mind?'),('user','Use the ground-floor lounge instead.')], 'Where should I go after the change of plan?', 'Go to the ground-floor lounge.', 'Go to the rooftop.'),
      ('request_offer', [('user','I cannot open the bottle with this injured wrist.'),('peer','I am sitting beside you.')], 'Would you like me to open it for you?', 'Yes, could you open it for me?', 'Yes, I will open it for you.'),
      ('unknown', [('peer','The coach confirmed that there will be training, but has not picked the starting time.'),('user','Okay, tell me once the time is decided.')], 'What time should I put in my calendar?', 'The starting time has not been decided yet.', 'Put 8 am in your calendar.'),
    ]
    rows=[dict(item_id=f'TURN-PROSPECTIVE-{i:03d}',family=fam,context=[t[1] for t in turns],
        speakers=[t[0] for t in turns],message=msg,positive=pos,negative=neg,
        criterion='Address the latest incoming message; preserve explicit roles, facts and constraints. Reference is one acceptable reply.',
        provenance='Assistant-authored synthetic prospective challenge, frozen before layout-v2 baseline/training; not independent human gold.')
        for i,(fam,turns,msg,pos,neg) in enumerate(tuples,1)]
    key=lambda r:json.dumps([r['context'],r['speakers'],r['message']])
    old={name:json.loads((BASE/'data'/f'{name}.json').read_text()) for name in ['train','validation','ordinary','legacy','diagnostic','challenge']}
    seen={key(r) for group in old.values() for r in group}
    assert not seen.intersection(key(r) for r in rows)
    for name in ['train','validation','ordinary','legacy','diagnostic']:save(OUT/'data'/f'{name}.json',old[name])
    save(OUT/'data/challenge.json',rows)
    recipe=copy.deepcopy(json.loads((BASE/'recipe.json').read_text()))
    recipe.update(experiment='sealion_turn_layout_v2',arms=['supervised'],
        objective='Same supervised loss/data/schedule as v1 supervised, with explicit history/latest-incoming/reply-speaker serialization; no margin term.',
        source_format='Earlier conversation: Me/Friend labels; separate Latest incoming message from Friend; Me (STANDARD) response boundary.',
        data_counts={**recipe['data_counts'],'challenge':16},
        provenance='Same192 targeted synthetic training and16 validation.16 new separately written prospective challenges frozen before layout-v2 baseline/training. Prior24 challenges are now development diagnostics, not fresh-test claims.',
        eval_policy='16 prospective synthetic challenge +1 reused user diagnostic +20 reused legacy validation. Base/old Singlish frozen. Select by same16 validation and ordinary50 retention rule; no challenge-based selection.',
        scope='A change in serialization jointly with training, not a decoder-architecture change or proof of independent human generalization.')
    recipe['data_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'data').glob('*.json'))}
    save(OUT/'recipe.json',recipe)
    print(json.dumps(dict(root=str(OUT),train=192,validation=16,prospective_challenge=16,arms=recipe['arms'])))


if __name__=='__main__':main()
