"""Multiple acceptable drafts with explicit context constraints.

Assistant-authored supplemental examples; original project records are untouched.
Gold alternatives/constraints are never supplied to runtime generation.
"""
import hashlib
import random

FAMILIES=['unknown_call','known_call','call_later','open_invitation','busy_invitation',
    'free_invitation','lend_request','borrow_offer','return_item','budget',
    'location','completed','unfinished','unseen_text','vague_help',
    'overwhelmed','postponed','thanks','open_preference','earliest_time']


def example(family,v,split):
    objects=['bike pump','camera battery','travel adapter','folding chair','drawing tablet','music stand','picnic mat']
    topics=['our presentation outline','the volunteer schedule','the flat viewing','our club registration','the rehearsal recording','the room booking','our travel plans']
    events=['practice','the workshop','a group meeting','the briefing','the study session','training','the exhibition']
    places=['the library entrance','the station exit','the sports hall','the cafe counter','the studio door','the canteen entrance','the bookshop']
    days=['Tuesday','Friday','Monday','Thursday','Wednesday','Saturday','Sunday']
    early=['2 pm','10 am','5 pm','9 am','3 pm','1 pm','4 pm'][v]
    late=['4 pm','noon','7 pm','11 am','6 pm','5 pm','8 pm'][v]
    obj,topic,event,place=objects[v],topics[v],events[v],places[v]
    busy,free=days[v],days[(v+2)%7]
    context=[];message='';pairs=[];bad='';criterion='';reason='contradiction'
    if family=='unknown_call':
        context=[('user',f'Could we have a quick call after {event}?'),('peer','I am occupied at the moment.'),('user','Okay, we can talk another time.')]
        message=['What did you want to talk about?','What were you hoping to discuss?','What is the call about?'][v%3]
        pairs=[('Just wanted to catch up a little. There is no rush.','Just wanted to catch up a bit lah, no rush.'),
            ("I wanted your opinion on something. I'll explain over text first.","Wanted your opinion on something. I text you first can?"),
            ("I thought we could chat for a bit. We can message instead if that is easier.","Thought we could chat a bit. Message also can if easier.")]
        bad='Why do you want to call me?';reason='speaker_reversal'
        criterion='User initiated the call. A newly proposed purpose or text-first action is allowed; do not reverse who called or invent an earlier agreement.'
    elif family=='known_call':
        context=[('user',f'The only thing I wanted to call you about was {topic}.'),('peer',f'I can talk after {late}.')]
        message='What did you want to discuss?'
        pairs=[(f'I wanted to discuss {topic}.',f'Wanted to discuss {topic} lah.'),
            (f'It was about {topic}. I can send the details over text first.',f'About {topic}. I text you the details first can?'),
            (f'Just {topic}. We can leave the call until later.',f'Just {topic}. Later then call also can.')]
        bad='The call was only about my gym membership.'
        criterion=f'Preserve the explicitly stated sole call purpose: {topic}.'
    elif family=='call_later':
        context=[('user',f'I cannot talk before {late}.'),('peer',f'I am heading to {event}.')]
        message=f'Can you call me at {early}?'
        pairs=[(f'I cannot call that early. Could we talk after {late}?',f'That early cannot leh. After {late} can?'),
            (f'I am only free after {late}. I can text you first.',f'Only free after {late} leh. I text you first.'),
            (f'I cannot make {early}. Would tomorrow be easier?',f'{early} cannot lah. Tomorrow easier anot?')]
        bad=f'Yes, I can call you at {early}.';criterion=f'User cannot call before {late}; proposed later actions are allowed.'
    elif family=='open_invitation':
        message=f'Would you like to join us for {event} on {free}?'
        pairs=[(f'Sure, what time on {free}?',f'Can, {free} what time ah?'),
            ("I'll pass this time, but thanks for inviting me.","This time I pass lah, thanks for asking."),
            ('Sounds interesting. Where are you meeting?', 'Sounds interesting leh. Meet where ah?')]
        bad='We already agreed yesterday that I would organise it.';reason='invented_shared_past'
        criterion='No user preference or prior agreement is supplied. Acceptance, decline or asking details are all valid proposed drafts.'
    elif family=='busy_invitation':
        context=[('user',f'I have a shift on {busy}; I cannot attend anything that day. I am free on {free}.')]
        message=f'Can you join {event} on {busy}?'
        pairs=[(f'I cannot make {busy}. Could we do {free}?',f'{busy} cannot leh. {free} can?'),
            (f'I have a shift on {busy}, so I will sit this one out.',f'{busy} got shift, this time I pass lah.'),
            (f'I cannot attend on {busy}. Could you send me an update afterwards?',f'{busy} I cannot go. Afterwards update me can?')]
        bad=f'Yes, I am free on {busy} and will attend.';criterion=f'Do not accept an event on explicitly unavailable {busy}.'
    elif family=='free_invitation':
        context=[('user',f'I am free on {free}. I have a shift on {busy}.')]
        message=f'Would {free} work for {event}?'
        pairs=[(f'Yes, {free} works. What time?',f'Can, {free} works. What time ah?'),
            (f'{free} is fine. Could we meet at {place}?',f'{free} can. Meet at {place} can?'),
            (f'I am available on {free}. How long will it take?',f'{free} I free. How long ah?')]
        bad=f'I have a shift on {free} and am not available.';criterion=f'{free} is free; {busy} is busy. Preserve which day has the shift.'
    elif family=='lend_request':
        context=[('peer',f'I left my {obj} at home.'),('user',f'I have my {obj} with me.')]
        message=f'Could you lend me yours until tomorrow?'
        pairs=[(f'Sure, I can lend you my {obj}.',f'Can, I lend you my {obj}.'),
            (f'Could you return my {obj} by tomorrow evening?',f'My {obj} tomorrow evening return can?'),
            (f'Could you pick up my {obj} at {place}?',f'Come {place} collect my {obj} can?')]
        bad=f'Could you lend me your {obj} instead?';reason='speaker_reversal';criterion=f'Friend requests User-owned {obj}; User is the lender, Friend the borrower.'
    elif family=='borrow_offer':
        context=[('user',f'I left my {obj} at home.'),('peer',f'I have a spare {obj}.')]
        message='Would you like to borrow mine?'
        pairs=[(f'Yes please, could I borrow your {obj}?',f'Yes please, lend me your {obj} can?'),
            ('Thanks, but I can manage without it today.', 'Thanks ah, but today I can manage without it.'),
            (f'That would help. Where can I collect your {obj}?',f'That helps leh. Your {obj} collect where ah?')]
        bad=f'I will lend you my spare {obj}.';reason='speaker_reversal';criterion='Friend offers their spare; User may borrow, decline or arrange collection, not become lender.'
    elif family=='return_item':
        context=[('user',f'I still have the {obj} you lent me.'),('peer','I would like it back tomorrow.')]
        message='How should we arrange the return?'
        pairs=[(f'I can bring your {obj} to {place} tomorrow.',f'Tomorrow I bring your {obj} to {place} can?'),
            (f'Would you like to collect your {obj} from me tomorrow?',f'Tomorrow you collect your {obj} from me can?'),
            ('What time tomorrow would suit you?', 'Tomorrow what time suits you ah?')]
        bad=f'You need to return the {obj} I lent you.';reason='speaker_reversal';criterion='User borrowed Friend item and should return it; collection/drop-off suggestions are open.'
    elif family=='budget':
        cap,cheap,expensive=12+v,7+v,25+v
        context=[('user',f'My spending limit is ${cap}.'),('peer',f'The snack costs ${cheap}, and the meal costs ${expensive}.')]
        message='Which one would you like?'
        pairs=[(f'I will take the ${cheap} snack; it is within my budget.',f'I take the ${cheap} snack lah, within budget.'),
            (f'The ${expensive} meal is over my budget. Could we find somewhere cheaper?',f'${expensive} over my budget leh. Find cheaper place can?'),
            ('I will skip buying anything this time.', 'This time I skip buying anything lah.')]
        bad=f'The ${expensive} meal is within my ${cap} budget.';criterion=f'Do not assert {expensive}<={cap}; choosing the snack, declining or suggesting cheaper options are valid.'
    elif family=='location':
        elsewhere=places[(v+1)%7]
        context=[('user',f'I am at {place}.'),('peer',f'I am at {elsewhere}.')]
        message='Where are you now?'
        pairs=[(f'I am at {place}.',f'I at {place} now.'),
            (f'I am at {place}. Shall I wait here?',f'I at {place}. Wait here can?'),
            (f'I am still at {place}. I can come over to you.',f'Still at {place} leh. I can come find you.')]
        bad=f'I am at {elsewhere} with you already.';criterion=f'Current User location is {place}, not Friend location {elsewhere}; future movement may be proposed.'
    elif family=='completed':
        context=[('user',f'I have finished the notes for {event} and uploaded them.'),('peer','Let me check the folder.')]
        message='Are your notes done?'
        pairs=[('Yes, they are finished and uploaded.', 'Done and uploaded liao.'),
            ('Yes, they are done. Want me to send you the link?', 'Done liao. Want me send you the link?'),
            ('They are finished. Could you check whether you can access them?', 'Finished liao. You check can open anot?')]
        bad='No, I have not started the notes.';criterion='Notes are explicitly finished/uploaded; do not reverse completion.'
    elif family=='unfinished':
        context=[('user',f'I am still writing the notes for {event}; they are not finished.')]
        message='Can you send the finished notes now?'
        pairs=[('They are not finished yet. I can send them when they are ready.', 'Not finished yet leh. Ready then I send.'),
            ('I am still working on them. Would a draft help first?', 'Still doing leh. Send you draft first can?'),
            ('They are not ready yet. Which part do you need first?', 'Not ready yet leh. Which part you need first?')]
        bad='I have already finished and sent all the notes.';criterion='Unfinished notes cannot be claimed finished/sent; offering a draft or future action is allowed.'
    elif family=='unseen_text':
        context=[('peer',f'I am writing an introduction for {event}.')]
        message='Can you tell me whether the paragraph is too long?'
        pairs=[('Send me the paragraph and I will take a look.', 'Send me the paragraph, I take a look.'),
            ('Sure, paste it here and we can trim it together.', 'Can, paste here then we trim together.'),
            ('Could you send it along with the word limit?', 'Send the paragraph and word limit can?')]
        bad='Yes, your paragraph is definitely too long.';reason='unsupported_judgment';criterion='No paragraph is supplied. Offer a useful next action, not a claim to have assessed unseen content.'
    elif family=='vague_help':
        context=[('peer',f'I am working on the exercises before {event}.')]
        message='I am stuck on one question. Can you help?'
        pairs=[('Send me the question and I will try to help.', 'Send the question, I try help you.'),
            ('Which step are you stuck on?', 'Which step stuck ah?'),
            ('Show me what you have tried and we can work through it.', 'Show me what you tried, we work through together.')]
        bad='The answer to that question is definitely 42.';reason='invented_answer';criterion='Question is absent; propose help or ask useful details rather than inventing its answer.'
    elif family=='overwhelmed':
        context=[('peer',f'I have to prepare for {event} and I am feeling overwhelmed.')]
        message='I do not know where to start.'
        pairs=[('Want to break it into smaller steps together?', 'Want break into smaller steps together anot?'),
            ('Would a short break help before you start?', 'Take a short break first can?'),
            ('Which part feels most urgent? I can help you plan.', 'Which part most urgent ah? I help you plan.')]
        bad='You told me everything was already finished.';reason='invented_shared_past';criterion='Respond supportively; no fabricated prior completion claim.'
    elif family=='postponed':
        context=[('peer',f'We are meeting for {event}.'),('user','Okay, see you there.')]
        message='The start has been pushed back by half an hour.'
        pairs=[('Thanks for letting me know. I will adjust my plans.', 'Thanks for telling me ah, I adjust my plans.'),
            ('What is the updated start time?', 'New start time what ah?'),
            ('No worries. Shall we grab a drink while we wait?', 'No worries lah. While waiting get a drink can?')]
        bad='Great, it now starts half an hour earlier.';criterion='A delay is later, not earlier; original clock time is unspecified.'
    elif family=='thanks':
        context=[('peer',f'I was having trouble preparing for {event}.'),('user','I can help with the checklist.')]
        message='Thanks for helping me today!'
        pairs=[('You are welcome!', 'No problem lah!'),
            ('Glad I could help. Anything else you want to check?', 'Glad can help. Anything else want check?'),
            ('Happy to help. Hope it goes smoothly!', 'Happy to help lah. Hope all goes smoothly!')]
        bad='You promised to pay me yesterday for helping.';reason='invented_shared_past';criterion='Acknowledge thanks or offer more help; no invented payment agreement.'
    elif family=='open_preference':
        context=[('peer',f'We can get a drink after {event}.'),('user','I do not want matcha today.')]
        message='What would you like to drink instead?'
        pairs=[('How about a coffee?', 'Coffee can?'),('I think I will get water.', 'I think I get water lah.'),('Maybe a hot tea. What are you having?', 'Maybe hot tea lah. You drinking what?')]
        bad='I only want matcha today.';criterion='User rejected matcha; other newly proposed preferences are valid.'
    elif family=='earliest_time':
        context=[('user',f'I can only leave {event} after {late}.'),('peer',f'I can wait at {place}.')]
        message=f'Can we meet at {early}?'
        pairs=[(f'I cannot make {early}. Could we meet after {late}?',f'{early} cannot leh. After {late} can?'),
            (f'I can only leave after {late}. Would tomorrow be easier?',f'Only can leave after {late} leh. Tomorrow easier anot?'),
            (f'That is too early for me. Could we message first?', 'Too early for me leh. Message first can?')]
        bad=f'Yes, I will be there at {early}.';criterion=f'Earliest departure is after {late}; do not accept earlier {early}.'
    else:raise ValueError(family)
    row=dict(item_id=f'CREATIVE-{split}-{family}-{v}',family=family,split=split,
        context=[t for s,t in context],speakers=[s for s,t in context],message=message,
        alternatives=[dict(english=e,singlish=l) for e,l in pairs],negative=bad,negative_reason=reason,
        criterion=criterion,scope='prospective_creative_'+split,
        provenance='Assistant-authored supplemental examples and audit; not team-authored or independent human gold.')
    audit(row);return row


def audit(row):
    assert len(row['context'])==len(row['speakers']) and all(s in {'user','peer'} for s in row['speakers'])
    assert len(row['alternatives'])==3
    for style in ['english','singlish']:
        strings=[r[style].strip() for r in row['alternatives']]
        assert all(strings) and len(set(strings))==3
    assert row['negative'].strip() and row['negative'] not in [a['english'] for a in row['alternatives']]
    assert all('information not available' not in a['english'].lower() for a in row['alternatives'])


def dataset():
    result={'train':[example(f,v,'train') for f in FAMILIES for v in range(5)],
        'validation':[example(f,5,'validation') for f in FAMILIES],
        'challenge':[example(f,6,'challenge') for f in FAMILIES]}
    # All variants of a context, including targets, belong to exactly one split.
    key=lambda r:(tuple(r['context']),tuple(r['speakers']),r['message'])
    sets={s:{key(r) for r in rows} for s,rows in result.items()}
    assert all(len(sets[s])==len(rows) for s,rows in result.items())
    assert not sets['train']&sets['validation'] and not sets['train']&sets['challenge'] and not sets['validation']&sets['challenge']
    return result


def schedule(ordinary_count,seed=12081,cycles=12):
    rng=random.Random(seed);visits=[0]*100;ordinary=list(range(ordinary_count));rng.shuffle(ordinary);cursor=0;plan=[]
    for cycle in range(cycles):
        ids=list(range(100));rng.shuffle(ids)
        for start in range(0,100,4):
            main=[]
            for _ in range(4):
                if cursor==ordinary_count:rng.shuffle(ordinary);cursor=0
                main.append(ordinary[cursor]);cursor+=1
            chosen=ids[start:start+4];variants=[visits[i]%3 for i in chosen]
            for i in chosen:visits[i]+=1
            plan.append(dict(ordinary=main,targeted=chosen,variants=variants))
    assert all(n==12 for n in visits) and len(plan)==300
    return plan


def case_seed(ident):return int(hashlib.sha256(ident.encode()).hexdigest()[:8],16)


def normalize_reply(text):return ' '.join(text.casefold().split())
