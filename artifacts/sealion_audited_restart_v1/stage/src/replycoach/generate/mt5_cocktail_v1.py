"""Auditable text-to-text auxiliary tasks for the mT5 cocktail pilot.

The synthetic ledger is training/evaluation supervision only. Runtime generation
receives just the conversation, never the ledger or reference answer.
"""
import hashlib
import json
import random

ARMS = {'baseline': (0., 0.), 'facts': (1/6, 0.),
        'answerability': (0., 1/6), 'both': (1/6, 1/6)}
FIELDS = ['room', 'amount_owed', 'bottle_colour', 'location',
          'call_topic', 'available_day', 'collector', 'start_time']


def value(field, index):
    return {
        'room': lambda: f'K{index+200}',
        'amount_owed': lambda: str(index+17),
        'bottle_colour': lambda: ['light', 'dark', 'pale', 'bright'][index//8%4]+' '+['blue','green','orange','purple','red','yellow','pink','grey'][index%8],
        'location': lambda: ['east','west','north','south'][index%4]+' lobby, floor '+str(index//4+1),
        'call_topic': lambda: ['the room booking','the group presentation','our travel plans','the club registration','the assignment deadline','the equipment return','our practice schedule','the shared expenses'][index%8],
        'available_day': lambda: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][index%7],
        'collector': lambda: ['Mira','Danial','Noor','Priya','Wei Lin','Hafiz','Sara','Jun'][index%8],
        'start_time': lambda: f'{8+index//60%14:02d}:{index%60:02d}',
    }[field]()


def statement(field, text):
    return {
        'room': 'I booked room {}.', 'amount_owed': 'I owe you ${}.',
        'bottle_colour': 'My water bottle is {}.', 'location': 'I am at the {}.',
        'call_topic': 'I wanted to call you about {}.',
        'available_day': 'I am free on {}.', 'collector': '{} is collecting my badge.',
        'start_time': 'My appointment starts at {}.',
    }[field].format(text)


def question(field, owner, alternate=False):
    # The incoming message is always spoken by Friend/peer.
    if owner not in {'user', 'peer'}:
        raise ValueError(owner)
    you, your = ('you', 'your') if owner == 'user' else ('I', 'my')
    if not alternate:
        return {
            'room': f'Which room did {you} book?',
            'amount_owed': f'How much did {you} say {you} owe '+('me?' if owner=='user' else 'you?'),
            'bottle_colour': f'What colour is {your} water bottle?',
            'location': f'Where did {you} say '+('you are?' if owner=='user' else 'I am?'),
            'call_topic': 'What did '+you+' want to call '+('me' if owner=='user' else 'you')+' about?',
            'available_day': f'Which day did {you} say '+('you are' if owner=='user' else 'I am')+' free?',
            'collector': f'Who is collecting {your} badge?',
            'start_time': f'What time does {your} appointment start?',
        }[field]
    return {
        'room': f'Remind me of the room {you} booked?',
        'amount_owed': 'What amount did '+you+' say '+you+' owe '+('me?' if owner=='user' else 'you?'),
        'bottle_colour': f'Remind me what colour {your} water bottle is?',
        'location': f'What location did {you} mention '+('you are' if owner=='user' else 'I am')+' at?',
        'call_topic': 'What was the topic '+you+' wanted to phone '+('me' if owner=='user' else 'you')+' about?',
        'available_day': 'On which day did '+you+' mention being available?',
        'collector': f'Remind me who is picking up {your} badge?',
        'start_time': f'When did {you} say {your} appointment begins?',
    }[field]


def required_fact(row):
    """Derive supervision from the authored ledger; never used for inference."""
    if row.get('self_contained'):
        return 'none|none|not_required'
    matching = [f for f in row['ledger'] if (f['owner'],f['field']) == (row['owner'],row['field'])]
    if len(matching)>1:
        raise ValueError('Ambiguous evidence')
    return '|'.join([row['owner'],row['field'],matching[0]['value'] if matching else 'unknown'])


def answer_label(row):
    return 'ANSWERABLE' if row.get('self_contained') or not required_fact(row).endswith('|unknown') else 'MISSING'


def audit_case(row):
    assert len(row['context']) == len(row['speakers'])
    assert all(s in {'user','peer'} for s in row['speakers'])
    for f in row['ledger']:
        index=f['turn']
        text=row['message'] if index==len(row['context']) else row['context'][index]
        speaker='peer' if index==len(row['context']) else row['speakers'][index]
        assert f['owner'] in {speaker,'shared'} and f['value'] in text
    assert row['fact_target']==required_fact(row)
    assert row['answer_target']==answer_label(row)


def auxiliary_source(row, task):
    if task not in {'FACTS','ANSWERABILITY'}:
        raise ValueError(task)
    turns=[f'{"You" if s=="user" else "Friend"}: {t}' for s,t in zip(row['speakers'],row['context'],strict=True)]
    turns.append('Friend: '+row['message'])
    instruction = ('Extract the needed fact as owner|field|value; use unknown if missing and none|none|not_required if no prior fact is needed.'
        if task=='FACTS' else 'Can this message be answered using the conversation, the message itself or basic knowledge? Output ANSWERABLE or MISSING.')
    return task+' '+instruction+' | '+' | '.join(turns)+' | Result:'


def make_cases(split, variants):
    offset={'train':0,'validation':200,'challenge':400}[split]
    rows=[]
    for v in range(variants):
        for fi,field in enumerate(FIELDS):
            owner='user' if v%2==0 else 'peer'
            other='peer' if owner=='user' else 'user'
            wanted=value(field,offset+v)
            other_value=value(field,offset+v+3)
            assert wanted!=other_value
            for state in ['supported','other_owner','irrelevant']:
                events=[dict(speaker=other,text=statement(field,other_value),field=field,value=other_value)] if state!='irrelevant' else []
                if state=='supported':
                    events.append(dict(speaker=owner,text=statement(field,wanted),field=field,value=wanted))
                notice=hashlib.sha256(f'{split}-{v}-{field}'.encode()).hexdigest()[:6]
                events.append(dict(speaker='peer',text=f'The notice on the wall has label {notice}.'))
                random.Random(v*31+fi).shuffle(events)
                row=dict(item_id=f'COCKTAIL-{split}-{v}-{field}-{state}',group_id=f'{split}-{v}-{field}',
                    family=field,state=state,owner=owner,field=field,
                    context=[e['text'] for e in events],speakers=[e['speaker'] for e in events],
                    message=question(field,owner,split!='train'),scope=split,
                    ledger=[dict(owner=e['speaker'],field=e['field'],value=e['value'],turn=i) for i,e in enumerate(events) if 'field' in e])
                row.update(fact_target=required_fact(row),answer_target=answer_label(row))
                audit_case(row);rows.append(row)
    control_count={'train':32,'validation':16,'challenge':8}[split]
    for i in range(control_count):
        if i%4==0:
            val=f'{10+i//4:02d}:{offset%60+7:02d}'
            message=f'The invitation says the event starts at {val}. What time is stated in the invitation?'
            ledger=[dict(owner='shared',field='start_time',value=val,turn=0)]
            row=dict(owner='shared',field='start_time',ledger=ledger)
        else:
            message={1:f'What is {offset+i+2} plus 3?',2:'Thank you for helping me carry the bags.',3:'Would you like to get lunch together?'}[i%4]
            # Vary an irrelevant prior turn so these aren't duplicate inputs.
            row=dict(owner='none',field='none',ledger=[],self_contained=True)
        context=[] if i%4==0 else [f'The noticeboard has label N{offset+i}.']
        row.update(item_id=f'COCKTAIL-{split}-self-{i}',group_id=f'{split}-self-{i}',family='self_contained',state='latest_fact' if i%4==0 else 'self_contained',
            context=context,speakers=['user']*len(context),message=message,scope=split)
        row.update(fact_target=required_fact(row),answer_target=answer_label(row))
        audit_case(row);rows.append(row)
    keys=[json.dumps([r['context'],r['speakers'],r['message']]) for r in rows]
    assert len(set(keys))==len(keys)
    return rows


def training_schedule(primary_count, auxiliary_count, seed=1, steps=784):
    def stream(count, extra):
        cycle=0
        while True:
            ids=list(range(count));random.Random(seed+extra+cycle).shuffle(ids)
            yield from ids;cycle+=1
    main,facts,answer=stream(primary_count,0),stream(auxiliary_count,10000),stream(auxiliary_count,20000)
    return [dict(main=[next(main) for _ in range(12)],facts=[next(facts) for _ in range(2)],answerability=[next(answer) for _ in range(2)]) for _ in range(steps)]
