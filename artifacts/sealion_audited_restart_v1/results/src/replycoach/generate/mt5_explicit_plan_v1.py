"""Explicit context notes passed into a separately trained reply generator.

Annotation functions below produce supervised labels/diagnostic oracles only.
Production-like inference receives a freely generated planner string instead.
"""
import re
import random
from replycoach.generate.mt5_matched_v1 import source


def planner_source(row):
    conversation=source(row,'STANDARD')[len('STANDARD '):]
    return 'CONTEXT_PLAN '+conversation+' | Summarize facts, roles, constraints and open choices:'


def planned_reply_source(row,notes):
    if not isinstance(notes,str):raise TypeError('Explicit note string required')
    conversation=source(row,'STANDARD')[len('STANDARD '):]
    return ('PLANNED_REPLY '+conversation+' | Context notes: '+(notes.strip() or '(none)')+
        ' | Suggest an English reply. Respect the conversation if notes conflict; new proposals are allowed:')


def creative_notes(row):
    """Assistant labels using source context/family, never reference replies."""
    family=row['family'];turns=row['context'];user=[t for t,s in zip(turns,row['speakers'],strict=True) if s=='user'];peer=[t for t,s in zip(turns,row['speakers'],strict=True) if s=='peer']
    def extract(pattern,text):
        m=re.search(pattern,text);assert m,(family,pattern,text)
        return m.group(1)
    if family=='unknown_call':
        facts='User initiated a call; Friend is occupied.';rules='Keep caller direction; no purpose has been stated.';open_='Propose a new intention, chatting later or texting first.'
    elif family=='known_call':
        topic=extract(r'only thing I wanted to call you about was (.+)\.',user[0])
        facts='User only wanted to call about '+topic+'.';rules='Preserve that stated purpose.';open_='Explain the purpose or offer to text details.'
    elif family=='call_later':
        earliest=extract(r'before (.+)\.',user[0]);facts='User cannot talk before '+earliest+'.';rules='Do not accept an earlier call.';open_='Propose a later call or text.'
    elif family in {'busy_invitation','free_invitation'}:
        if family=='busy_invitation':
            busy=extract(r'shift on ([^;]+);',user[0]);free=extract(r'free on ([^.]+)\.',user[0])
        else:
            busy=extract(r'shift on ([^.]+)\.',user[0]);free=extract(r'free on ([^.]+)\.',user[0])
        facts=f'User has a shift on {busy}; User is free on {free}.';rules=f'Do not describe {busy} as free or {free} as the shift day.';open_='Accept compatible plans, decline or propose another arrangement.'
    elif family=='open_invitation':
        facts='Friend invites User; User preference is unstated.';rules='Do not invent an earlier agreement.';open_='Acceptance, decline or asking details are all possible drafts.'
    elif family=='lend_request':
        obj=extract(r'I have my (.+) with me\.',user[0]);facts=f'User has the {obj}; Friend asks to borrow it.';rules='User is lender; Friend is borrower.';open_='Lend it, set return conditions or arrange collection.'
    elif family=='borrow_offer':
        obj=extract(r'I have a spare (.+)\.',peer[0]);facts=f'Friend offers a spare {obj}; User left theirs at home.';rules='Friend is lender; User would borrow.';open_='Accept, decline or ask where to collect it.'
    elif family=='return_item':
        obj=extract(r'I still have the (.+) you lent me\.',user[0]);facts=f'User borrowed the {obj} from Friend; Friend wants it back tomorrow.';rules='User returns Friend item.';open_='Propose drop-off, collection or ask a suitable time.'
    elif family=='budget':
        cap=int(extract(r'\$(\d+)',user[0]));prices=[int(x) for x in re.findall(r'\$(\d+)',peer[0])];cheap,cost=prices
        assert cheap<=cap<cost
        facts=f'User spending limit ${cap}; snack ${cheap}; meal ${cost}.';rules='Meal exceeds the stated limit; snack fits.';open_='Choose snack, decline purchase or propose a cheaper option.'
    elif family=='location':
        here=extract(r'I am at (.+)\.',user[0]);there=extract(r'I am at (.+)\.',peer[0]);facts=f'User is at {here}; Friend is at {there}.';rules='Preserve current locations.';open_='State User location, wait or propose moving later.'
    elif family=='completed':
        facts='User has finished and uploaded the notes.';rules='Do not claim the notes are unfinished.';open_='Confirm completion, offer the link or ask whether access works.'
    elif family=='unfinished':
        facts='User is still writing the notes; they are unfinished.';rules='Do not claim finished notes are ready or sent.';open_='Offer a draft, ask which part is needed or send when ready.'
    elif family=='unseen_text':
        facts='Friend asks whether a paragraph is too long; the paragraph is absent.';rules='No assessment of the unseen paragraph is established.';open_='Ask for the paragraph or word limit and offer to help.'
    elif family=='vague_help':
        facts='Friend is stuck on an unspecified exercise question.';rules='The question and its answer are not supplied.';open_='Offer help, ask for the question or ask what they tried.'
    elif family=='overwhelmed':
        facts='Friend feels overwhelmed preparing for an event.';rules='No prior completion is established.';open_='Offer support, a break or help planning smaller steps.'
    elif family=='postponed':
        facts='Friend says the event starts half an hour later.';rules='Delay means later, not earlier; original clock time is unspecified.';open_='Acknowledge, ask the updated time or suggest something while waiting.'
    elif family=='thanks':
        facts='Friend thanks User for helping.';rules='No payment agreement is established.';open_='Acknowledge warmly or offer further help.'
    elif family=='open_preference':
        facts='User does not want matcha today; Friend asks for another drink preference.';rules='Preserve rejection of matcha.';open_='Propose a new drink preference or ask Friend preference.'
    elif family=='earliest_time':
        earliest=extract(r'after (.+)\.',user[0]);facts='User can only leave after '+earliest+'.';rules='Do not accept meeting before User can leave.';open_='Propose a later time, another day or messaging first.'
    else:raise ValueError(family)
    return f'FACTS: {facts} RULES: {rules} OPEN: {open_}'


def ledger_notes(row):
    facts='; '.join(f"{f['owner']} {f['field']}={f['value']}" for f in row['ledger']) or 'No relevant prior fact is supplied'
    if row.get('self_contained'):
        rules='Latest message can receive an ordinary reply; empty history is allowed.'
    else:
        present=any((f['owner'],f['field'])==(row['owner'],row['field']) for f in row['ledger'])
        rules=f"Question concerns {row['owner']} {row['field']}; "+('that fact is supplied.' if present else 'that fact is unspecified; do not claim an established answer.')
    return f'FACTS: {facts}. RULES: {rules} OPEN: A useful new proposal or clarification is allowed.'


def diagnostic_notes(row):
    ident=row['item_id']
    fixed={
        'COCKTAIL-NATURAL-01':'User unavailable Monday and free Wednesday. Friend proposes rehearsal on Monday at six. Do not accept unavailable Monday.',
        'COCKTAIL-NATURAL-02':'User busy Wednesday and free Monday. Friend proposes rehearsal on Monday at six. Monday is compatible; preserve which day is busy.',
        'COCKTAIL-NATURAL-03':'Friend charger stopped working. User has their own charger. Friend asks to borrow User charger until tomorrow. User lends; Friend borrows.',
        'COCKTAIL-NATURAL-04':'User charger stopped working. Friend offers their own charger until tomorrow. Friend lends; User borrows.',
        'COCKTAIL-NATURAL-05':'User has eight dollars for lunch. Bowl costs sixteen dollars. Cost exceeds available budget. Propose a feasible arrangement, cheaper choice or decline; do not claim sixteen fits eight.',
        'COCKTAIL-NATURAL-06':'User has twenty dollars for lunch. Bowl costs sixteen dollars. Cost fits budget. Preferences are open.',
        'COCKTAIL-NATURAL-07':'User completed and emailed slides this morning. Friend cannot find them and requests resend now. Preserve prior completion; a resend or checking address is possible.',
        'COCKTAIL-NATURAL-08':'User has not finished slides. Presentation is tomorrow morning. Friend requests completed slides now. Do not claim they are already ready; offer a draft or future action.',
        'COCKTAIL-NATURAL-09':'User wanted to call about repairing bicycle brake cable. Friend is in a tutorial. They can talk after class. Preserve purpose and Friend current unavailability.',
        'COCKTAIL-NATURAL-10':'User initiated a call. Friend is in a tutorial; they can talk after class. Purpose unspecified. A new intention or text-first proposal is allowed; keep caller direction.',
        'COCKTAIL-NATURAL-11':'User can leave lab only after four. Cafe closes at six. Friend asks to meet at half past three. Requested time is earlier than User can leave.',
        'COCKTAIL-NATURAL-12':'User can leave lab after three. Cafe closes at six. Friend asks to meet at half past three. Proposed time is compatible; preferences remain open.',
        'USER-CALL-001':'User initiated the call. Friend is not free and asks what User wants to call about. Purpose unspecified. Propose a plausible intention or text-first action; do not reverse caller or insist on an immediate call.',
        'CALL-WITH-PURPOSE':'User wanted to call about the assignment deadline. Friend is not free now. Preserve that topic and current unavailability; offer later call or text.',
    }
    return fixed.get(ident)


def generator_schedule(ordinary_count,new_count=100,seed=4601,steps=300):
    def stream(count,offset):
        cycle=0
        while True:
            ids=list(range(count));random.Random(seed+offset+cycle).shuffle(ids)
            yield from ids;cycle+=1
    old,new=stream(ordinary_count,0),stream(new_count,10000);visits=[0]*new_count;plan=[]
    for step in range(steps):
        chosen=[next(new) for _ in range(2)];refs=[visits[i]%3 for i in chosen]
        for i in chosen:visits[i]+=1
        plan.append(dict(ordinary=[next(old) for _ in range(6)],new=chosen,refs=refs,
            notes_origin=['gold' if (step+j)%2==0 else 'predicted' for j in range(2)]))
    assert all(n==6 for n in visits)
    return plan
