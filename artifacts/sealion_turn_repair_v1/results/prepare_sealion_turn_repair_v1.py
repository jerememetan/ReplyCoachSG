"""Freeze a latest-turn repair study; generated targets are explicitly synthetic."""
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/sealion_turn_repair_v1'
TEAM_SHA = '03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text() != text:
        raise FileExistsError('Frozen study file differs: ' + str(path))
    path.write_text(text)


def row(ident, family, turns, message, positive, negative, criterion):
    return dict(item_id=ident, family=family, context=[t[1] for t in turns],
        speakers=[t[0] for t in turns], message=message, positive=positive,
        negative=negative, criterion=criterion,
        provenance='Assistant-authored diagnostic/training target; not human gold.')


def training():
    rows = []
    objects = ['camera battery', 'bike pump', 'drawing tablet', 'travel adapter', 'music stand', 'picnic mat']
    topics = ['the missing survey responses', 'our presentation outline', 'the flat viewing',
        'the broken lab equipment', 'our volunteer schedule', 'the rehearsal recording']
    events = ['practice', 'training', 'rehearsal', 'revision', 'the workshop', 'the briefing']
    places = ['the library entrance', 'the station exit', 'the sports hall', 'the cafe counter', 'the studio door', 'the bus interchange']
    other_places = ['the courtyard', 'the auditorium', 'the foyer', 'the canteen', 'the computer room', 'the bookshop']
    times = [('2 pm','4 pm'),('10 am','noon'),('5 pm','7 pm'),('9 am','11 am'),('3 pm','6 pm'),('1 pm','5 pm')]
    days = [('Tuesday','Friday'),('Monday','Thursday'),('Wednesday','Saturday'),('Thursday','Sunday'),('Friday','Monday'),('Saturday','Tuesday')]
    for v in range(6):
        obj, topic, event, place, other = objects[v], topics[v], events[v], places[v], other_places[v]
        early, late = times[v]; free, busy = days[v]
        cheap, cap, costly = 8+v, 15+v, 32+v
        for q in range(2):
            cases = []
            def add(family, turns, messages, positives, negatives, criterion):
                cases.append(row(f'TURN-TRAIN-{family}-{v}-{q}', family, turns, messages[q], positives[q], negatives[q], criterion))
            add('call_known_reason', [('user',f'I need to ask you about {topic}. I can talk after {late}.'),
                ('peer','I am in the middle of something.'),('user','Could we have a quick call when you are free?')],
                ['What do you want to discuss on the call?','When can you talk?'],
                [f'I wanted to ask you about {topic}. I can text you first.',f'I can talk after {late}.'],
                ['Could you call me immediately?',f'I wanted to ask you about {topic}.'],
                'Answer latest purpose/time question, rather than repeat the earlier request.')
            add('call_unspecified_reason', [('user','Can I give you a ring?'),('peer',f'I am occupied with {event}.'),('user','Okay, we can talk later.')],
                ['What did you want to talk to me about?','Could you send a message instead?'],
                ["I'll send you a message explaining what I wanted to discuss.","Sure, I'll explain it over text."],
                ['Why do you want to call me?','Can you ring me right now?'],
                'Maintain who initiated the call; a text-first draft is allowed when the purpose is unspecified.')
            add('help_specificity', [('user',f'I need help with {topic}.'),('peer','What exactly do you need?'),('user',f'I need you to check the figures, not rewrite the whole thing. I can send it after {late}.')],
                ['Which part should I look at?','When will you send it?'],
                ['Could you check the figures?',f'I can send it after {late}.'],
                ['Please rewrite the whole thing.',f'I can send it before {early}.'],
                'Answer the latest requested detail and retain the stated task/time.')
            add('completion_query', [('user',f'I have packed the {obj}, but I have not printed the handout.'),('peer','Thanks for preparing those.')],
                [f'Have you packed the {obj}?','Have you printed the handout?'],
                [f'Yes, I have packed the {obj}.','I have not printed the handout yet.'],
                [f'I have not packed the {obj} yet.','Yes, I have printed the handout.'],
                'Preserve the queried item and its completion polarity.')
            add('availability_query', [('user',f'I am free on {free}, but I have a shift on {busy}.'),('peer',f'I want to arrange {event}.')],
                [f'Can you make it on {free}?',f'Can you make it on {busy}?'],
                [f'Yes, {free} works for me.',f'I have a shift on {busy}. Could we do {free}?'],
                [f'I cannot make {free}.',f'Yes, {busy} works for me.'],
                'Use the queried day, not a generic acceptance.')
            add('borrow_direction', [('user',f'I borrowed your {obj}. You borrowed my headphones.'),('peer','We should return our things.')],
                [f'Who needs to return the {obj}?','Who needs to return the headphones?'],
                [f'I need to return your {obj}.','You need to return my headphones.'],
                [f'You need to return my {obj}.','I need to return your headphones.'],
                'Bind the returner to the borrowed object, retaining first/second person.')
            add('offer_recipient', [('peer',f'I left my {obj} at home.'),('user',f'I have a spare {obj}, and I do not need any more.')],
                [f'Could you lend me your {obj}?','Should I bring you a spare instead?'],
                [f'Sure, I can lend you my {obj}.',f'No need, I already have a spare {obj}.'],
                [f'Could you lend me your {obj}?',f'Yes, I need you to bring me a spare {obj}.'],
                'Respond to the incoming request/offer from the correct perspective.')
            add('budget_choice', [('user',f'My budget is ${cap}.'),('peer',f'The snack costs ${cheap}; the meal costs ${costly}.')],
                ['Which option is within your budget?',f'Want the ${costly} meal?'],
                [f'The ${cheap} snack is within my budget.',f'The ${costly} meal is over my budget. I will take the ${cheap} snack.'],
                [f'The ${costly} meal is within my budget.',f'Yes, I can afford the ${costly} meal.'],
                'Respect the explicit budget and distinguish a question from an offer.')
            add('location_perspective', [('user',f'I am waiting at {place}.'),('peer',f'I am still at {other}.')],
                ['Where are you waiting?','Where did I say I was?'],
                [f'I am waiting at {place}.',f'You said you were at {other}.'],
                [f'I am waiting at {other}.',f'You said you were at {place}.'],
                'Bind each location to its actual speaker.')
            add('time_constraint', [('user',f'I cannot leave work until {late}.'),('peer',f'I was thinking about {event}.')],
                [f'Can we meet at {early}?','What is the earliest you can leave?'],
                [f'I cannot meet at {early}; I cannot leave until {late}.',f'I can leave at {late}.'],
                [f'Yes, {early} is fine.',f'I can leave at {early}.'],
                'Preserve explicit availability instead of accepting an incompatible time.')
            add('decline_reason', [('user',f'I have a work shift during {event}.'),('peer',f'Can you join {event}?'),('user','Sorry, I cannot join.')],
                ['Why can you not join?','Would moving it to a day off help?'],
                ['I have a work shift at that time.','Yes, a day when I am not working would help.'],
                [f'Can you join {event}?','No, my work schedule is not the reason.'],
                'Explain the earlier refusal when asked why; do not repeat the invitation.')
            add('latest_correction', [('user',f'Let us meet at {other}.'),('peer','That place is closed.'),('user',f'Okay, change it to {place}.')],
                ['Where are we meeting now?','Which place was closed?'],
                [f'We are meeting at {place}.',f'{other[0].upper()+other[1:]} was closed.'],
                [f'We are meeting at {other}.',f'{place[0].upper()+place[1:]} was closed.'],
                'Distinguish the updated plan from the rejected plan.')
            add('future_not_done', [('user',f'I have not checked the {obj} yet. I plan to check it after {event}.'),('peer','Please remember to check it.')],
                [f'Have you checked the {obj} already?','When are you planning to check it?'],
                [f'Not yet. I will check it after {event}.',f'I plan to check it after {event}.'],
                [f'Yes, I have checked the {obj}.',f'I checked it before {event}.'],
                'Do not convert a future plan into a completed action.')
            add('peer_update', [('user',f'Could you check whether my {obj} is ready?'),('peer','I checked it. It is ready to collect.')],
                ['I can collect it for you if you want.','Can you collect it yourself?'],
                ['Thanks, could you collect it for me?','Yes, I can collect it myself.'],
                ['I will collect it for you.',f'Could you check whether the {obj} is ready?'],
                'Answer the new offer/request instead of replaying the earlier question.')
            add('known_unknown_details', [('user',f'I know {event} is on {free}, but its starting time has not been announced.'),('peer','I am updating my calendar.')],
                ['What day is it?','What time does it start?'],
                [f'It is on {free}.','The starting time has not been announced yet.'],
                [f'It is on {busy}.',f'It starts at {early}.'],
                'Return known facts, and do not invent an explicitly unknown detail.')
            add('question_after_request', [('user',f'Can you send me the notes for {event}?'),('peer','I have them, but the file is large.'),('user','You can use a download link.')],
                ['Which notes do you need?', 'How should I send them to you?'],
                [f'I need the notes for {event}.','Could you send me a download link?'],
                ['How should I send the notes to you?', 'Could you tell me which notes you need?'],
                'Answer a follow-up clarification while preserving who requested the notes.')
            rows.extend(cases)
    assert len(rows) == 192
    return rows


def authored(split):
    # Each tuple is independently written; no slot-substituted training template
    # is reused. Skills intentionally overlap. These are not human test labels.
    cases = {
      'validation': [
        ('purpose', [('user','I noticed the totals in our charity account do not match.'),('peer','I am cooking now.'),('user','Could we speak once you finish?')], 'What would you like to go through?', 'I wanted to go through the mismatched totals in our charity account.', 'Could you call me this instant?'),
        ('purpose_unknown', [('user','Hey, got a moment for a voice chat?'),('peer','I am tied up.'),('user','No problem, we can speak another time.')], 'Can you first tell me what this is regarding?', "I'll explain what I wanted to discuss in a message first.", 'What did you want to call me about?'),
        ('clarification', [('user','I need a hand editing the captions, not the video.'),('peer','Sure, I can help after dinner.')], 'What exactly should I edit?', 'Please edit the captions.', 'Please edit the video.'),
        ('status', [('user','I saved a draft of the email but have not sent it.'),('peer','The coordinator is waiting.')], 'Has the coordinator received your email?', 'I have only saved a draft; I have not sent it yet.', 'Yes, I already sent the email.'),
        ('return', [('peer','I still have the ruler you lent me.'),('user','I need it back for my drawing.')], 'Should you pass it to me or should I pass it to you?', 'You should pass my ruler back to me.', 'I should pass your ruler back to you.'),
        ('available', [('user','I am on duty throughout Wednesday. Thursday is clear.'),('peer','I want to meet up this week.')], 'Would Wednesday work for you?', 'I am on duty on Wednesday. Could we meet on Thursday?', 'Wednesday works for me.'),
        ('budget', [('user','I can spend at most $16 on this outing.'),('peer','The ticket alone is $28.')], 'Shall I buy that ticket for you?', 'That ticket is over my $16 budget. Please do not buy it for me.', 'Yes, buy the $28 ticket for me.'),
        ('correction', [('peer','Shall we gather at the fountain?'),('user','The fountain area is fenced off; use the west lobby instead.'),('peer','Okay, west lobby it is.')], 'Where should I head to now?', 'Head to the west lobby.', 'Head to the fountain.'),
        ('offer', [('user','My laptop battery is almost flat.'),('peer','I have a compatible charger with me.')], 'Want to borrow my charger?', 'Yes, could I borrow your charger?', 'Yes, you can borrow my charger.'),
        ('explanation', [('user','I have lost my voice, so I cannot join the voice meeting.'),('peer','Are you feeling unwell?')], 'Why can you not join by voice?', 'I have lost my voice.', 'Can you join the voice meeting?'),
        ('unknown', [('peer','The organiser has confirmed Sunday, but has not given us a venue.'),('user','Okay, I will wait for the details.')], 'Where is it being held?', 'The venue has not been confirmed yet.', 'It is being held in the main hall.'),
        ('recipient', [('user','I can pick up the badges for both of us.'),('peer','That would save me a trip.')], 'Can you get mine as well?', 'Yes, I can pick up your badge too.', 'Can you pick up my badge too?'),
        ('focus', [('user','The tutorial is online at 11 am. The lab is in D214 at 3 pm.'),('peer','I mixed up the two sessions.')], 'Which room is the lab in?', 'The lab is in D214.', 'The tutorial is online at 11 am.'),
        ('timing', [('user','I finish my medical appointment at 5 pm and cannot leave earlier.'),('peer','I would like some company for dinner.')], 'Can you meet at 4 pm?', 'I cannot meet at 4 pm; my appointment finishes at 5 pm.', 'Yes, I can meet at 4 pm.'),
        ('acknowledge', [('user','Please tell me when the parcel arrives.'),('peer','It has arrived and I put it in your locker.')], 'Can you check your locker later?', 'Yes, I will check my locker. Thanks for putting it there.', 'Has the parcel arrived yet?'),
        ('request_detail', [('user','Could you review my bibliography?'),('peer','Yes, send it over.'),('user','I will share the document.')], 'What do you want me to check in it?', 'Could you check the bibliography?', 'What do you want me to send you?'),
      ],
      'challenge': [
        ('call_topic', [('user','There is a clash between my exam and the camp briefing.'),('peer','I am boarding a train.'),('user','Can we talk when you get off?')], 'Before that, what is this about?', 'I wanted to discuss the clash between my exam and the camp briefing.', 'Can you call me right now?'),
        ('call_topic', [('user','I need your advice on the lease renewal.'),('peer','In a meeting, cannot talk.'),('user','Okay, just let me know when you have a moment.')], 'What did you need advice about?', 'I wanted your advice on the lease renewal.', 'What advice did you want from me?'),
        ('call_unspecified', [('user','Can we have a quick chat over the phone?'),('peer','Sorry, I am at a family dinner.'),('user','Another time is fine.')], 'Could you tell me the topic first?', "I'll send a message explaining what I wanted to talk about first.", 'Why do you want to call me?'),
        ('clarification', [('user','Please help me check the names on the guest list.'),('peer','I can do that tonight.'),('user','Thank you, that would help.')], 'Do you need me to check the names or design the invitations?', 'Please check the names on the guest list.', 'Please design the invitations.'),
        ('clarification', [('peer','I have copies of the route map and the price list.'),('user','Could you send me the route map?'),('peer','Sure.')], 'Which one did you ask for again?', 'I asked for the route map.', 'I asked for the price list.'),
        ('already_answered', [('user','Has the poster been printed?'),('peer','Yes, it is printed and waiting at reception.'),('user','Great, thanks.')], 'Can you pick it up when you arrive?', 'Yes, I can pick up the poster from reception.', 'Has the poster been printed yet?'),
        ('completion', [('user','I filled in the application, but have not pressed submit.'),('peer','The deadline is tonight.')], 'Is your application submitted already?', 'Not yet. I have filled it in but have not submitted it.', 'Yes, my application is submitted already.'),
        ('completion', [('user','I have sent the slides. I still need to send the speaker notes.'),('peer','I am assembling the final folder.')], 'Have you sent the speaker notes?', 'I have not sent the speaker notes yet.', 'Yes, I have sent the speaker notes.'),
        ('ownership', [('peer','I borrowed your calculator last week.'),('user','I will need it for my quiz tomorrow.')], 'Who should bring the calculator back?', 'You should bring my calculator back to me.', 'I should bring your calculator back to you.'),
        ('ownership', [('user','The cooler box in my car is yours; you lent it to me for the picnic.'),('peer','We should sort out the picnic equipment.')], 'Am I returning that box to you or are you returning it to me?', 'I am returning your cooler box to you.', 'You are returning my cooler box to me.'),
        ('availability', [('user','I am looking after my niece on Saturday and cannot go out. Sunday is okay.'),('peer','I want to visit the museum.')], 'Shall I book us in for Saturday?', 'I cannot go on Saturday. Could we go on Sunday?', 'Yes, book Saturday for us.'),
        ('availability', [('user','My flight lands at 9 pm, so I cannot reach town before then.'),('peer','I will be around town that evening.')], 'Can you be there at 7 pm?', 'I cannot be there at 7 pm; my flight only lands at 9 pm.', 'Yes, I can be there at 7 pm.'),
        ('amount', [('user','I have $14 left to spend today.'),('peer','The two options cost $9 and $26.')], 'Which option can you afford?', 'I can afford the $9 option.', 'I can afford the $26 option.'),
        ('amount', [('user','Please keep my share under $20.'),('peer','The activity will cost $35 each.')], 'Should I reserve your place?', 'That exceeds my $20 limit. Please do not reserve my place.', 'Yes, reserve my place for $35.'),
        ('correction', [('user','I first said the north gate, but that gate is locked.'),('peer','What should we do?'),('user','Meet at the taxi stand instead.')], 'Where are we meeting after the change?', 'We are meeting at the taxi stand.', 'We are meeting at the north gate.'),
        ('location', [('peer','I am at the ticket office.'),('user','I am waiting by the lift on level two.')], 'Where can I find you?', 'I am by the lift on level two.', 'I am at the ticket office.'),
        ('request_offer', [('user','My hands are full carrying these boxes.'),('peer','I have a trolley with me.')], 'Shall I bring the trolley over to you?', 'Yes, please bring the trolley over to me.', 'Yes, I will bring the trolley over to you.'),
        ('request_offer', [('peer','I forgot my umbrella.'),('user','I have an extra one in my bag.')], 'Could I use your spare?', 'Sure, you can use my spare umbrella.', 'Could I use your umbrella?'),
        ('reason', [('user','I cannot eat peanuts because I am allergic.'),('peer','I was going to bring peanut cookies.')], 'Why do you need to avoid them?', 'I am allergic to peanuts.', 'Could you bring me peanut cookies?'),
        ('reason', [('user','My microphone is broken, so I cannot speak in the online discussion.'),('peer','I can see you are connected.')], 'Why are you only typing?', 'My microphone is broken.', 'Why are you only typing?'),
        ('unknown', [('peer','They have confirmed that the event is happening, but are still choosing a date.'),('user','Let me know when they decide.')], 'What day is it on?', 'The date has not been decided yet.', 'It is on Friday.'),
        ('known_detail', [('user','The interview is in room F309 at 10:40. My briefing is in G112 at 2 pm.'),('peer','I want to walk over with you.')], 'Which room is your interview in?', 'My interview is in F309.', 'My interview is in G112.'),
        ('future', [('user','The recording is still on my laptop. I will upload it after I get home.'),('peer','I need to download it later.')], 'Is it uploaded now?', 'Not yet. I will upload it after I get home.', 'Yes, I have uploaded it already.'),
        ('speaker_plan', [('peer','I can bring dessert if you bring the drinks.'),('user','Okay, I will bring the drinks.')], 'What am I supposed to bring?', 'You are bringing dessert.', 'You are bringing the drinks.'),
      ]
    }[split]
    return [row(f'TURN-{split.upper()}-{i:03d}', fam, turns, msg, pos, neg,
        'Answer the latest incoming message while preserving explicit actors, facts and constraints; reference is one acceptable reply.')
        for i,(fam,turns,msg,pos,neg) in enumerate(cases,1)]


def main():
    assert sha(ROOT/'data/dev/train.jsonl') == TEAM_SHA
    source = ROOT/'data/training_expansion/generated600_repair_v1/combined_train.jsonl'
    original = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    ordinary = [dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],
        speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],
        positive=r['replies']['standard']) for r in original]
    legacy = [json.loads(line) for line in (ROOT/'data/dev/validation.jsonl').read_text().splitlines() if line.strip()]
    legacy = [dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],
        speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],positive=r['replies']['standard']) for r in legacy]
    groups = dict(train=training(), validation=authored('validation'),challenge=authored('challenge'))
    diagnostic = dict(item_id='USER-CALL-001',family='user_failure_diagnostic',
        context=['Can u call me now?','Eh i not so free eh?','Ohh, then when can we call?'],
        speakers=['user','peer','user'],message='I am not sure, what do you want to call me about',
        criterion='Respond as the person who initiated the call. Do not reverse roles. Purpose is unspecified, so do not treat a specific invented purpose as established fact.')
    key = lambda r: json.dumps([r['context'],r['speakers'],r['message']],ensure_ascii=False)
    previous = sum([json.loads((ROOT/p).read_text()) for p in [
        'data/evaluation/sealion_full_rewrite_v1/inputs.json','data/evaluation/restored_pipeline_v1/inputs.json','data/evaluation/context_pairs_v1/inputs.json']],[])
    sets = {name:{key(r) for r in rows} for name,rows in groups.items()}
    for name,rows in groups.items():assert len(sets[name])==len(rows)
    assert not (sets['train'] & sets['validation'] or sets['train'] & sets['challenge'] or sets['validation'] & sets['challenge'])
    assert not set.union(*sets.values()).intersection(key(r) for r in [*ordinary,*legacy,*previous,diagnostic])
    for r in groups['train']+groups['validation']+groups['challenge']:
        assert r['positive']!=r['negative'] and len(r['context'])==len(r['speakers'])
    # Existing validation is reused only for retention/selection and regression
    # reporting, never treated as a new held-out score.
    for name,rows in dict(**groups,ordinary=ordinary,legacy=legacy,diagnostic=[diagnostic]).items():save(DEST/'data'/f'{name}.json',rows)
    matched=[r['id'] for r in original if re.search(r'call|what.*about|why|what.*(need|want)|what.*happen',r['message']['text'],re.I)]
    audit=dict(project_conversations=len(original),broad_latest_message_regex_matches=len(matched),matched_ids=matched,
        interpretation='Rough literal search, not proof of semantic coverage. Inspection of these six finds no direct purpose-of-call follow-up target.',
        prompt_speaker_mapping='user=You, peer=Friend; latest incoming always Friend; original user failure matches this contract.',
        preprocessing='No truncation; causal prompt tokens masked; EOS retained. New rows have explicit paired latest-question variants.',
        original_team_sha256=TEAM_SHA,project_source_sha256=sha(source),locked_test_read=False)
    save(DEST/'audit.json',audit)
    recipe=dict(experiment='sealion_turn_repair_v1',account='hm.teoh.2024',
        source_bundle='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1',
        initial_english_sha256='55a7f02626c08894bfdc4b44f68dacf86e8f4edafb7aa6bd78aead9162698913',
        frozen_singlish_sha256='6ea7066deda4f4e9e5f4a9e070d46427536a0dac05d5a416864e97c2dc3f5db2',
        data_counts={name:len(rows) for name,rows in dict(**groups,ordinary=ordinary,legacy=legacy,diagnostic=[diagnostic]).items()},
        provenance='192 assistant-authored targeted training dialogues;16 independently written synthetic validation cases;24 separately written challenge cases frozen before baseline/training. Skills overlap; not independent human test or project locked test.',
        arms=['supervised','contrastive'],seed=8411,optimizer_steps=120,learning_rate=2e-5,
        weight_decay=.01,clip_grad_norm=1.,replay_per_step=1,targeted_per_step=3,
        objective='0.25 ordinary CE +0.25 sum3 targeted positive CEs +weight*relu(.2+positiveCE-negativeCE); weight0/.1. Same examples, initial weights and per-step dropout seed.',
        snapshots=[0,40,80,120],
        selection='Select on16 synthetic validation positives only, subject to ordinary50 validation token CE<=1.05*initial; step0 eligible. Lower targeted positive mean CE wins. Challenge and user outputs cannot select checkpoints.',
        eval_policy='Generate baseline/selected English and own-English-conditioned Singlish on24 challenge +1 user diagnostic +20 deterministic old validation regression cases. Review intended action, factual fidelity and local naturalness separately. Contrast ranking is teacher-forced, not generated accuracy.',
        inference=dict(num_beams=4,no_repeat_ngram_size=3,max_new_tokens=64,total_token_limit=384),
        source_weights_unchanged=True,original_team_records_unchanged=True,default_changed=False,
        data_sha256={p.name:sha(p) for p in sorted((DEST/'data').glob('*.json'))})
    save(DEST/'recipe.json',recipe)
    print(json.dumps(dict(root=str(DEST),counts=recipe['data_counts'],audit=audit)))


if __name__ == '__main__':
    main()
