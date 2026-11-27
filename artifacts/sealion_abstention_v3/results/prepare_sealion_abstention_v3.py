"""Correct synthetic speaker references and add context-independent factual controls."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from replycoach.generate.sealion_abstention_v1 import audit_question_perspective, REFUSAL
PRIOR=ROOT/'artifacts/sealion_abstention_v2'
OUT=ROOT/'artifacts/sealion_abstention_v3'


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if p.exists() and p.read_text()!=text:raise FileExistsError(p)
    p.write_text(text)


def factual_controls(split):
    commons={
        'train':[('Is a carrot a vegetable?','Yes, a carrot is a vegetable.','yes'),
            ('Is a pencil a writing tool?','Yes, a pencil is a writing tool.','yes'),
            ('Is a dog an animal?','Yes, a dog is an animal.','yes'),
            ('Is a square a three-sided shape?','No, a square has four sides.','no'),
            ('Is a stone a living animal?','No, a stone is not a living animal.','no'),
            ('Is Tuesday a day of the week?','Yes, Tuesday is a day of the week.','yes'),
            ('Is an orange a tool?','No, an orange is a fruit.','no'),
            ('Is a bicycle a type of tree?','No, a bicycle is a vehicle.','no')],
        'validation':[('Is a banana a fruit?','Yes, a banana is a fruit.','yes'),
            ('Is a chair a piece of furniture?','Yes, a chair is furniture.','yes'),
            ('Is Wednesday a month?','No, Wednesday is a day of the week.','no'),
            ('Is a triangle a four-sided shape?','No, a triangle has three sides.','no'),
            ('Is a hammer a type of fruit?','No, a hammer is a tool.','no'),
            ('Is a bus a vehicle?','Yes, a bus is a vehicle.','yes'),
            ('Is an oak tree an animal?','No, an oak tree is a plant.','no'),
            ('Is a spoon a utensil?','Yes, a spoon is a utensil.','yes')]}[split]
    rows=[]
    for v in range(8):
        if split=='train':hour=10+v%6;minute=(v*7+5)%60;a=4+v;b=2+v%3;code='Hall S'+str(100+v*3)
        else:hour=16+v%6;minute=(v*11+3)%60;a=12+v;b=3+v%4;code='Room V'+str(600+v*5)
        value=f'{hour:02d}:{minute:02d}'
        examples=[('incoming_time',f'The invitation says the session begins at {value}. What start time does it give?',f'It gives {value}.',value),
            ('arithmetic',f'What is {a} plus {b}?',f'{a} plus {b} is {a+b}.',str(a+b)),
            ('incoming_room',f'The note lists {code} as the venue. Which venue is written in the note?',f'The venue is {code}.',code),
            ('common_knowledge',*commons[v])]
        for kind,message,answer,expected in examples:
            context=[] if v%2==0 else ['We have finished cleaning the fan.','The room looks tidier.','Okay, thank you.']
            rows.append(dict(item_id=f'SELF-FACT-{split}-{v}-{kind}',family='self_contained_factual',kind=kind,
                context=context,speakers=[] if not context else ['user','peer','user'],message=message,
                positive=answer,requires_refusal=False,reference_values=[[expected]],scope=split,
                layout='self_contained_facts' if split=='validation' else None,
                provenance='Assistant-authored context-independent factual control. Fact is in latest message or answer uses simple arithmetic/common knowledge.'))
    return rows


def main():
    train=copy.deepcopy(json.loads((PRIOR/'data/train.json').read_text()))
    changed=[]
    for row in train:
        old=row['message']
        if row.get('family') in {'meeting_room','amount_owed','item_colour','item_count','collector_name','file_name'}:
            text=re.sub(r'\bdid I say\b','did you say',old,flags=re.I)
            text=re.sub(r'\bI booked\b','you booked',text,flags=re.I)
            text=re.sub(r'\bI packed\b','you packed',text,flags=re.I)
            text=re.sub(r'\bI owe you\b','you owe me',text,flags=re.I)
            text=re.sub(r'\bmy water bottle\b','your water bottle',text,flags=re.I)
            row['message']=text
        if old!=row['message']:changed.append(row['item_id'])
        if row.get('family')=='call_topic' and row['requires_refusal'] is False:
            v=int(row['item_id'].split('-')[3])
            phrase=['I wanted to call you about','I wanted to ask you about','I need to talk to you about','I need your advice on'][v]
            row['context']=[t.replace('I wanted to call you about',phrase) for t in row['context']]
    assert len(changed)==144  #72 supported and72 missing variants; original790 skipped.
    assert all(audit_question_perspective(r) for r in train)
    train+=factual_controls('train')
    by_id={r['item_id']:r for r in train};control=[]
    for row in train:
        if row['requires_refusal']:
            restored=copy.deepcopy(by_id[row['supported_counterpart']]);restored['item_id']='CONTROL-'+row['item_id'];control.append(restored)
        else:control.append(copy.deepcopy(row))
    validation=copy.deepcopy(json.loads((PRIOR/'data/validation.json').read_text()))+factual_controls('validation')
    # Fresh paired histories, written before this correction run is evaluated.
    facts=[
        ('purpose','I wanted to ask you about arranging the study trip.','What was it you wanted to call me about?','I wanted to discuss arranging the study trip.',['study trip'],'Could we talk by phone when you are free?'),
        ('room','I booked room R842.','Which room did you book?','I booked room R842.',['R842'],'I am arranging the room booking.'),
        ('debt','I owe you $63.','How much did you say you owe me?','I owe you $63.',['63'],'We are settling the shared expenses.'),
        ('collector','Noor is collecting our keys.','Who is collecting our keys?','Noor is collecting our keys.',['Noor'],'We are organising the key collection.'),
        ('venue','We agreed on the east-wing lounge.','Where did we agree to meet?','At the east-wing lounge.',['east-wing lounge'],'We are arranging a place to meet.'),
        ('time','We agreed to meet at 13:55.','What time did we agree to meet?','We agreed on 13:55.',['13:55'],'We are arranging a meeting.'),
        ('reference','My reservation reference is DJ704.','What is your reservation reference?','My reference is DJ704.',['DJ704'],'I am looking through my reservation details.'),
        ('duration','We agreed that the call would last 25 minutes.','How long did we agree the call would last?','We agreed on 25 minutes.',['25'],'We are arranging a call.')]
    challenge=[]
    for i,(family,fact,message,answer,values,neutral) in enumerate(facts):
        for missing in [False,True]:
            challenge.append(dict(item_id=f'CORRECTED-TEST-{i}-'+('missing' if missing else 'supported'),family=family,
                context=[neutral if missing else fact,'I am busy for the next few minutes.','Okay, we can speak later.'],
                speakers=['user','peer','user'],message=message,positive=REFUSAL if missing else answer,
                requires_refusal=missing,reference_values=[] if missing else [values],scope='prospective_synthetic_v3',
                provenance='New assistant-authored paired history; not independent human benchmark.'))
    extras=[
        ('Would you like to get breakfast together this weekend?','Yes, that sounds good.',[]),
        ('Thank you for helping me carry those chairs upstairs.','You are welcome.',[]),
        ('What is 3 plus 4?','3 plus 4 is 7.',[['7','seven']]),
        ('Is a pear a fruit?','Yes, a pear is a fruit.',[['yes']]),
        ('The poster says the event starts at 18:45. What time does it state?','It states 18:45.',[['18:45']]),
        ('The note says the venue is A931. Which venue is listed?','The venue is A931.',[['A931']]),
        ('My budget is $15 and the ticket costs $10. Is the ticket within my budget?','Yes, it is within your budget.',[['yes']]),
        ('Would you like to borrow my spare umbrella?','Yes please, could I borrow it?',[])]
    for i,(message,answer,values) in enumerate(extras):
        challenge.append(dict(item_id=f'CORRECTED-SELF-{i}',family='self_contained',context=[],speakers=[],message=message,
            positive=answer,requires_refusal=False,reference_values=values,scope='prospective_synthetic_v3',
            provenance='Assistant-authored self-contained control; not unique human gold.'))
    legacy=json.loads((PRIOR/'data/legacy.json').read_text())
    key=lambda r:json.dumps([r['context'],r['speakers'],r['message'].strip().lower()])
    assert len(train)==1152 and len({key(r) for r in train})==1152
    assert sum(r['requires_refusal'] is True for r in train)==120
    assert len(validation)==128 and len(challenge)==24
    assert not {key(r) for r in train}&{key(r) for r in validation+challenge+legacy}
    assert not {key(r) for r in validation}&{key(r) for r in challenge}
    for name,rows in dict(train=train,control=control,validation=validation,challenge=challenge,legacy=legacy).items():save('data/'+name+'.json',rows)
    recipe=copy.deepcopy(json.loads((PRIOR/'recipe.json').read_text()))
    recipe.update(experiment='sealion_abstention_v3_role_corrected',training_slots_per_epoch=1152,
        unique_inputs=dict(restored_context_control=1032,selective_abstention=1152),
        optimizer_steps=288,snapshots=[0,72,144,288],unanswerable_fraction=120/1152,
        self_contained_controls=122,validation=dict(cases=128,underlying_prior_conversations=48,history_layouts=2,new_self_contained_factual_cases=32),
        challenge=dict(cases=24,missing=8,answerable=16),
        selection='Maximize worst balanced score over the two mixed answerability history layouts. Require over-refusal<=10%, lexical value matches>=own baseline-1 and no generation failures for EACH layout, including32 answerable-only factual controls. Answerable-only controls gate selection but do not enter the minimum balanced score.',
        initialization='Reset to untouched original English adapter; do not continue the V1/V2 adapters trained with erroneous speaker references.',
        hypothesis='Correct speaker-relative questions and teach abstention only when required facts are absent from BOTH history and the latest message; protect context-independent arithmetic/common knowledge.',
        matching='Correction run; same rank8 original source and LR1e-5,2epochs,batch8,seed9217.1152 slots instead of1120 due32 additional self-contained factual controls. No matched restored-context control rerun.',
        provenance='Earlier V1/V2 training had ambiguous/wrong first-person questions in six synthetic families, affecting72 supported rows; those models are diagnostic only, not promoted. Correct144 questions across supported/missing variants, add speaker-perspective tests, preserve all790 source records. Fresh24-case synthetic challenge frozen before this run; prior tests are development. No independent human or locked-test result.')
    recipe['data_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'data').glob('*.json'))}
    save('recipe.json',recipe)
    save('audit.json',dict(changed_question_ids=changed,potentially_wrong_supported_targets_in_old_data=72,
        original790_unchanged=train[:790]==json.loads((PRIOR/'data/train.json').read_text())[:790],
        original_team_sha256=json.loads((PRIOR/'audit.json').read_text())['original_team_sha256'],
        trained_refusal_rows=120,additional_self_contained_factual_controls=32,
        perspective_audit_passed=all(audit_question_perspective(r) for r in train),locked_test_read=False))
    print(json.dumps(dict(train=len(train),refusal=120,fraction=120/1152,validation=len(validation),challenge=len(challenge),corrected_questions=len(changed))))


if __name__=='__main__':main()
