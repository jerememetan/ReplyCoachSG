"""Balance evidence position and remove missingness/Update lexical shortcuts."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'artifacts/sealion_abstention_v1'
OUT=ROOT/'artifacts/sealion_abstention_v2'


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if p.exists() and p.read_text()!=text:raise FileExistsError(p)
    p.write_text(text)


def story(fact,neutral,position):
    if position==0:return [fact,'I am finishing something at the moment.','Okay, we can continue later.'],['user','peer','user']
    if position==1:return ['I am finishing something at the moment.',fact,'Okay, thanks for explaining.'],['peer','user','peer']
    return [neutral,'Okay, we can sort it out.',fact],['user','peer','user']


def main():
    train=copy.deepcopy(json.loads((PRIOR/'data/train.json').read_text()))
    neutral={
        'meeting_time':'We are arranging a meeting.', 'meeting_room':'I am sorting out our meeting room.',
        'agreed_day':'We are making plans for a picnic.', 'amount_owed':'We are sorting out some shared expenses.',
        'item_colour':'I left my water bottle behind.', 'item_count':'I packed apples for the outing.',
        'collector_name':'We are arranging to collect some badges.', 'call_topic':'Could we talk on the phone later?',
        'file_name':'I am getting an attachment ready.', 'meeting_place':'We are planning where to gather.'}
    by_id={r['item_id']:r for r in train}
    for row in train:
        if row.get('family') not in neutral:continue
        parts=row['item_id'].split('-');variant=int(parts[3]);kind=row['kind']
        if row['requires_refusal'] is False:
            fact=row['context'][-1].removeprefix('Update: ')
            if kind=='empty':row['context']=[fact];row['speakers']=['user']
            elif kind=='irrelevant':
                if variant%2:row['context']=['We were talking about cleaning the windows.',fact];row['speakers']=['peer','user']
                else:row['context']=[fact,'We were talking about cleaning the windows.'];row['speakers']=['user','peer']
            else:row['context'],row['speakers']=story(fact,neutral[row['family']],variant%3)
        elif kind=='empty':continue
        elif kind=='irrelevant':
            row['context']=[neutral[row['family']] if variant%2 else 'The car engine needs routine maintenance.'];row['speakers']=['user']
        else:row['context'],row['speakers']=story(neutral[row['family']],neutral[row['family']],variant%3)
    for row in train:
        if row.get('family')=='self_contained_reply' and row['kind']!='empty':
            row['context']=['We finished cleaning the electric fan.','It looks much better now.','Okay, thanks for helping.']
            row['speakers']=['user','peer','user']
    control=[]
    for row in train:
        if row['requires_refusal']:
            restored=copy.deepcopy(by_id[row['supported_counterpart']]);restored['item_id']='CONTROL-'+row['item_id'];control.append(restored)
        else:control.append(copy.deepcopy(row))
    original_validation=json.loads((PRIOR/'validation_amendment/validation.json').read_text())
    alternate=copy.deepcopy(original_validation)
    for row in original_validation:row['layout']='canonical'
    for row in alternate:
        row['item_id']='POSITION-'+row['item_id'];row['layout']='fact_earlier'
        if row['reference_values']:
            fact=row['context'][-1].removeprefix('Update: ')
            row['context'],row['speakers']=story(fact,'We are arranging the details.',0)
        elif row['requires_refusal']:
            row['context'],row['speakers']=story('We are arranging the details.','We are arranging the details.',0)
        elif row['context']:
            row['context'],row['speakers']=story('We were cleaning a fan.','We were cleaning a fan.',0)
    validation=original_validation+alternate
    # Eight matched known/unknown triples, plus four self-contained controls.
    fresh=[
        ('call_purpose','I wanted to discuss the shift swap for the charity booth.','What did you want to talk about on the call?','I wanted to discuss the shift swap for the charity booth.',['shift swap'],'Could we speak by phone when you have a moment?'),
        ('call_reason','I need your advice on choosing a dissertation adviser.','What did you need my advice about?','I need your advice on choosing a dissertation adviser.',['dissertation adviser'],'Can we have a chat on the phone?'),
        ('collection_time','We agreed to collect the equipment at 15:50.','What time did we agree to collect the equipment?','We agreed on 15:50.',['15:50'],'We are arranging to collect the equipment.'),
        ('booking_room','I reserved room H673 for our practice.','Which room did you reserve?','I reserved room H673.',['H673'],'I am sorting out the room reservation.'),
        ('amount_due','I owe you $47 for my share.','How much did you say you owe me?','I owe you $47.',['47'],'We are settling the shared expenses.'),
        ('collector','Janelle is picking up our wristbands.','Who is picking up our wristbands?','Janelle is picking them up.',['Janelle'],'We are sorting out the wristband collection.'),
        ('agreed_venue','We agreed to meet at the greenhouse entrance.','Where did we agree to meet?','At the greenhouse entrance.',['greenhouse entrance'],'We are planning to meet up.'),
        ('reference','My booking reference is BV853.','What is your booking reference?','My booking reference is BV853.',['BV853'],'I am looking through my booking details.')]
    challenge=[]
    for i,(family,fact,question,answer,aliases,unknown) in enumerate(fresh):
        for missing in [False,True]:
            context,speakers=story(unknown if missing else fact,unknown,0)
            challenge.append(dict(item_id=f'POSITION-TEST-{i:02d}-'+('missing' if missing else 'supported'),
                family=family,kind='mixed_turns',context=context,speakers=speakers,message=question,
                positive='Information not available.' if missing else answer,
                requires_refusal=missing,reference_values=[] if missing else [aliases],scope='prospective_synthetic_v2',
                provenance='Assistant-authored paired synthetic scenario frozen before v2 baseline/training; no independent human score.'))
    for i,(message,answer) in enumerate([
        ('I really appreciate you staying behind to pack the equipment.','You are welcome. Happy to help.'),
        ('Would you like me to bring you a hot drink?','Could you bring me a cup of tea?'),
        ('I finally completed the first draft of my dissertation!','Congratulations on finishing your draft!'),
        ('Could you wait while I finish locking the door?','Sure, take your time.')]):
        context,speakers=story('We were discussing the weather.','We were discussing the weather.',0)
        challenge.append(dict(item_id=f'POSITION-SELF-{i}',family='self_contained',kind='mixed_turns',
            context=context,speakers=speakers,message=message,positive=answer,requires_refusal=False,
            reference_values=[],scope='prospective_synthetic_v2',provenance='Assistant-authored self-contained control.'))
    key=lambda r:json.dumps([r['context'],r['speakers'],r['message'].strip().lower()])
    assert len(train)==1120 and len({key(r) for r in train})==1120
    assert sum(r['requires_refusal'] is True for r in train)==120
    assert len(challenge)==20 and len({key(r) for r in challenge})==20
    assert not {key(r) for r in challenge}&{key(r) for r in train+validation}
    legacy=json.loads((PRIOR/'data/legacy.json').read_text())
    for name,rows in dict(train=train,control=control,validation=validation,challenge=challenge,legacy=legacy).items():save('data/'+name+'.json',rows)
    recipe=copy.deepcopy(json.loads((PRIOR/'recipe.json').read_text()))
    recipe.update(experiment='sealion_abstention_v2_position_balanced',arms=['selective_abstention'],
        validation=dict(cases=96,underlying_conversations=48,layouts=2,missing_per_layout=15,answerable_per_layout=33),
        challenge=dict(cases=20,missing=8,answerable=12),
        selection='Maximize the worst generated answerability balanced score over two48-case validation layouts. Each layout must have over-refusal<=10%, supported-value matches>=its own baseline-1 and no generation failures. Step0 eligible; no challenge/user-based selection.',
        matching='Same source English adapter,1120 slot order, labels,2epochs/280updates/LR1e-5/seed9217 as v1 treatment. Changes are evidence placement, speaker interleaving and removal of missingness/Update cues. No new matched restored-context arm; compare to v1 and original with that limitation.',
        provenance='Position-balanced augmentation prompted by v1 transfer failures. Original main test/probes are now development diagnostics. Twenty new synthetic challenge rows frozen before v2 output inspection. No locked test or independent human benchmark.',
        hypothesis='Explicit abstention supervision must include supported facts earlier in a multi-turn history, not only the latest Update statement.')
    recipe['data_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'data').glob('*.json'))}
    save('recipe.json',recipe)
    save('audit.json',dict(original_team_sha256=json.loads((PRIOR/'audit.json').read_text())['original_team_sha256'],
        original_ordinary790_unchanged=train[:790]==json.loads((PRIOR/'data/train.json').read_text())[:790],
        training_reply_labels_unchanged=[r['positive'] for r in train]==[r['positive'] for r in json.loads((PRIOR/'data/train.json').read_text())],
        actual_user_diagnostic_not_in_training=True,locked_test_read=False))
    print(json.dumps(dict(train=1120,refusal=120,validation_views=96,underlying_validation=48,new_synthetic_challenge=20)))


if __name__=='__main__':main()
