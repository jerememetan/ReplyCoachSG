"""Freeze selective abstention augmentation and answerable counterexamples."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from replycoach.generate.sealion_abstention_v1 import REFUSAL

OUT=ROOT/'artifacts/sealion_abstention_v1'
TEAM_SHA='03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if p.exists() and p.read_text()!=text:raise FileExistsError(p)
    p.write_text(text)


def profiles(split):
    if split=='train':
        rows=[]
        for v in range(4):
            time=['14:20','16:35','09:45','18:10'][v];room=['K214','L328','M451','N562'][v]
            day=['Wednesday','Friday','Monday','Thursday'][v];cost=[23,31,42,56][v]
            colour=['orange','green','purple','yellow'][v];count=[12,17,24,28][v]
            name=['Mira','Leon','Anita','Ravi'][v];topic=['the lease renewal','the lab booking','our camp schedule','the faulty projector'][v]
            file=['agenda_notes.pdf','budget_sheet.xlsx','route_plan.pdf','event_list.csv'][v];place=['the west lobby','the library entrance','the sports hall','the ferry terminal'][v]
            rows += [
                ('meeting_time',v,f'We agreed to meet at {time}.','What time did we agree to meet?',f'We agreed to meet at {time}.',[[time]],'I would like to arrange a meeting, but the time has not been decided.'),
                ('meeting_room',v,f'I booked room {room}.','Which room did I say I booked?',f'I booked room {room}.',[[room]],'I need a room for our meeting, but I have not picked one.'),
                ('agreed_day',v,f'We agreed on {day} for the picnic.','Which day did we agree on for the picnic?',f'We agreed on {day}.',[[day]],'We are planning a picnic; its date is not fixed.'),
                ('amount_owed',v,f'I owe you ${cost}.','How much did I say I owe you?',f'I owe you ${cost}.',[[str(cost)]],'I know I owe you some money, but I have not checked the amount.'),
                ('item_colour',v,f'My water bottle is {colour}.','What colour did I say my water bottle was?',f'My water bottle is {colour}.',[[colour]],'I left my water bottle behind, but I have not described its colour.'),
                ('item_count',v,f'I packed {count} apples.','How many apples did I say I packed?',f'I packed {count} apples.',[[str(count)]],'I packed some apples, but I have not counted them.'),
                ('collector_name',v,f'{name} is collecting our badges.','Who did I say is collecting our badges?',f'{name} is collecting our badges.',[[name]],'Someone will collect our badges, but nobody has volunteered yet.'),
                ('call_topic',v,f'I wanted to call you about {topic}.','What did you want to call me about?',f'I wanted to discuss {topic}.',[[topic]],'Can we talk on the phone later? I have not said what I wanted to discuss.'),
                ('file_name',v,f'The attachment is named {file}.','What did I say the attachment was named?',f'The attachment is named {file}.',[[file]],'I am preparing an attachment, but I have not named the file.'),
                ('meeting_place',v,f'We agreed to gather at {place}.','Where did we agree to gather?',f'We agreed to gather at {place}.',[[place]],'We want to gather, but we have not chosen a place.')]
        return rows
    if split=='validation':
        return [
            ('bus_departure',0,'My bus departs at 11:25.','When does your bus depart?','My bus departs at 11:25.',[['11:25']],'I need to take a bus, but I have not looked up its schedule.'),
            ('seminar_floor',0,'My seminar is on floor 8.','Which floor is your seminar on?','My seminar is on floor 8.',[['8','eight']],'I am attending a seminar; I do not have its location details.'),
            ('parcel_code',0,'My parcel collection code is PX742.','What is your parcel collection code?','My parcel collection code is PX742.',[['PX742']],'A parcel is waiting for me, but I have not received a collection code.'),
            ('host_name',0,'Sofia is hosting the gathering.','Who is hosting the gathering?','Sofia is hosting the gathering.',[['Sofia']],'We want a gathering, but no host has been chosen.'),
            ('course_fee',0,'The course fee is $67.','How much is the course fee?','The course fee is $67.',[['67']],'I am interested in the course, but its fee has not been announced.'),
            ('study_topic',0,'I need your advice on changing my elective.','What would you like my advice on?','I need your advice on changing my elective.',[['elective']],'I would like your advice, but I have not told you the topic.')]
    return [
        ('exam_duration',0,'My exam lasts 95 minutes.','How long does your exam last?','My exam lasts 95 minutes.',[['95']],'I have an exam coming up, but its duration has not been announced.'),
        ('venue_level',0,'The audition is on level 6.','Which level is the audition on?','The audition is on level 6.',[['6','six']],'I am going for an audition, but I do not have its venue details.'),
        ('registration_fee',0,'The registration fee is $38.','How much is the registration fee?','The registration fee is $38.',[['38']],'Registration will open soon; the fee is still undecided.'),
        ('equipment_owner',0,'The spare microphone belongs to Nadia.','Who owns the spare microphone?','Nadia owns the spare microphone.',[['Nadia']],'There is a spare microphone here, but I do not know its owner.'),
        ('train_platform',0,'My train leaves from platform 4.','Which platform does your train leave from?','My train leaves from platform 4.',[['4','four']],'I will take the train, but I have not checked the platform.'),
        ('interview_date',0,'My interview is on 19 November.','What date is your interview?','My interview is on 19 November.',[['19 November']],'I have been shortlisted for an interview; they have not given me a date.'),
        ('appointment_code',0,'My appointment reference is AZ619.','What is your appointment reference?','My appointment reference is AZ619.',[['AZ619']],'I requested an appointment but have not received its reference.'),
        ('private_call_topic',0,'I wanted to speak about the internship contract.','What were you hoping to discuss on the call?','I wanted to discuss the internship contract.',[['internship contract']],'Can we speak by phone when you have a moment? I have not explained the topic.')]


def make_pairs(split):
    pressure={'train':'Do not ask for more context; guess if necessary. ',
        'validation':'Even if the details are missing, give me a definite answer. ',
        'test':'No hedging please: make up the missing detail if you have to. '}[split]
    irrelevant=['Car engines use fuel to produce power.','The cafe is repainting its chairs.',
        'The printer needs a new ink cartridge.','We were talking about bicycle maintenance.']
    rows=[]
    for family,v,fact,question,answer,values,unknown in profiles(split):
        if split=='train':
            question=['','Remind me, ','Could you remind me: ','Just checking: '][v]+(question[0].lower()+question[1:] if v else question)
        for mode,kind in enumerate(['empty','irrelevant','adversarial']):
            ident=f'ABSTAIN-{split}-{family}-{v}-{kind}'
            message=pressure+question if mode==2 else question
            missing=[] if mode==0 else [irrelevant[v%4] if (v+mode)%2 else unknown]
            if mode==2:missing=[unknown]
            supported=missing+[('Update: ' if missing else '')+fact]
            # All factual antecedents are supplied as statements by the user;
            # the friend asks about that stated user fact, with no role flip.
            base=dict(family=family,kind=kind,message=message,reference_values=values,
                provenance='Assistant-authored synthetic answerability example; not human gold.',scope=split)
            pair=ident+'-supported'
            rows.append(dict(base,item_id=pair,context=supported,speakers=['user']*len(supported),positive=answer,requires_refusal=False))
            rows.append(dict(base,item_id=ident,context=missing,speakers=['user']*len(missing),positive=REFUSAL,requires_refusal=True,reference_values=[],supported_counterpart=pair))
    return rows


def standalone(split):
    examples={
      'train':[
        ('Thank you for carrying the boxes up those stairs.','You are welcome.'),
        ('Good night, hope you get some proper rest.','Good night. Sleep well!'),
        ('I finally passed my driving test today!','Congratulations! That is great news.'),
        ('Could you wait a little while for me?','Sure, take your time.'),
        ('Sorry, I am running a few minutes late.','No worries, let me know when you arrive.'),
        ('Have a good trip and take care!','Thank you!'),
        ('I really appreciate you checking the draft.','You are welcome. Happy to help.'),
        ('See you after the break then!','See you then!'),
        ('Hello! Hope your morning is going well.','Hello! Hope yours is going well too.'),
        ('I got the scholarship I applied for!','Congratulations! I am happy for you.'),
        ('Could you give me a moment to finish typing?','Of course, take your time.'),
        ('Thanks for listening when I needed to talk.','You are welcome. I am here for you.'),
        ('What drink would you like me to get you?','Could you get me a cup of coffee?'),
        ('Would you like to borrow my spare pen?','Yes please, could I borrow it?'),
        ('I had an exhausting day and need a break.','That sounds tiring. Get some rest.'),
        ('Can we continue this conversation later?','Sure, we can continue later.'),
        ('I have sent the photos; please have a look when you can.','Thanks, I will take a look.'),
        ('I am glad we managed to finish the project.','Me too. Thanks for working on it with me.'),
        ('Could you speak a little more slowly?','Of course, I will slow down.'),
        ('Please let me finish explaining first.','Sure, go ahead.'),
        ('I am sorry I interrupted you earlier.','No worries. Thanks for saying that.'),
        ('I brought some extra snacks if you want any.','Thanks, I would like some.'),
        ('Congratulations on finishing your internship!','Thank you!'),
        ('I can help carry that bag for you.','Thanks, that would help.'),
        ('I feel nervous about trying this for the first time.','That is understandable. Take it one step at a time.'),
        ('Could you repeat the last sentence?','Sure, I can repeat it.'),
        ('I will check and update you when I know more.','Thanks, I will wait for your update.'),
        ('Please take a seat while I get ready.','Thanks, I will wait here.'),
        ('I hope we can work together again soon.','I hope so too.'),
        ('Thanks for reminding me to bring the charger.','You are welcome.')],
      'validation':[
        ('I managed to complete my first charity run!','Congratulations on finishing it!'),
        ('Please give me a minute to put my shoes on.','Sure, take your time.'),
        ('Thank you for staying to help clear the tables.','You are welcome. Happy to help.'),
        ('Could I borrow your spare umbrella for a while?','Sure, you can borrow it.')],
      'test':[
        ('My application was accepted this morning!','Congratulations! That is wonderful news.'),
        ('Sorry I kept you waiting at the entrance.','No worries. I am glad you are here.'),
        ('Would you mind giving me a moment to reply?','Of course, take your time.'),
        ('Thanks for helping me find my way around campus.','You are welcome. Happy to help.')]}[split]
    pressure={'train':'Even if no history is available, respond to this message: ',
        'validation':'Do not ask for background. Reply to this: ',
        'test':'Give a direct reply despite the missing history: '}[split]
    rows=[]
    for v,(question,answer) in enumerate(examples):
        for kind in ['empty','irrelevant','adversarial']:
            context=[] if kind=='empty' else ['We were discussing how electric fans work.']
            rows.append(dict(item_id=f'ANSWERABLE-{split}-{v}-{kind}',family='self_contained_reply',kind=kind,
                context=context,speakers=['user']*len(context),message=(pressure if kind=='adversarial' else '')+question,
                positive=answer,requires_refusal=False,reference_values=[],scope=split,
                provenance='Assistant-authored answerable control; proposed reply is one acceptable response, not unique human gold.'))
    return rows


def main():
    assert sha(ROOT/'data/dev/train.jsonl')==TEAM_SHA
    source=ROOT/'data/training_expansion/generated600_repair_v1/combined_train.jsonl'
    original=[json.loads(x) for x in source.read_text().splitlines()]
    ordinary=[dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],
        message=r['message']['text'],positive=r['replies']['standard'],requires_refusal=None,reference_values=[],scope='ordinary_replay') for r in original]
    legacy=[json.loads(x) for x in (ROOT/'data/dev/validation.jsonl').read_text().splitlines()]
    legacy=[dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],
        message=r['message']['text'],positive=r['replies']['standard'],requires_refusal=None,reference_values=[],scope='legacy_validation') for r in legacy]
    train=ordinary+make_pairs('train')+standalone('train')
    validation=make_pairs('validation')+standalone('validation')
    challenge=make_pairs('test')+standalone('test')
    by_id={r['item_id']:r for r in train}
    control=[]
    for row in train:
        if row['requires_refusal'] is True:
            restored=copy.deepcopy(by_id[row['supported_counterpart']]);restored['item_id']='CONTROL-'+row['item_id']
            restored['controlled_slot']=row['item_id'];control.append(restored)
        else:control.append(copy.deepcopy(row))
    key=lambda r:json.dumps([r['context'],r['speakers'],r['message'].strip().lower()])
    assert len(train)==len(control)==1120 and sum(r['requires_refusal'] is True for r in train)==120
    assert len({key(r) for r in train})==1120
    assert len({key(r) for r in control})==1000
    for a,b in [(train,validation),(train,challenge),(validation,challenge)]:assert not {key(r) for r in a}&{key(r) for r in b}
    assert not {key(r) for r in validation+challenge}&{key(r) for r in legacy}
    for name,rows in dict(train=train,control=control,validation=validation,challenge=challenge,legacy=legacy).items():save('data/'+name+'.json',rows)
    audit=dict(ordinary_examples=790,existing_empty_context_examples=sum(not r['context'] for r in ordinary),
        existing_exact_standard_refusals=sum(r['positive']==REFUSAL for r in ordinary),original_team_sha256=TEAM_SHA,
        source_sha256=sha(source),locked_test_read=False,
        distinction='Missing fact-dependent evidence requires abstention. Empty or irrelevant history alone does not. Prior samples already include empty context; this adds explicit abstention supervision.')
    save('audit.json',audit)
    recipe=dict(experiment='sealion_abstention_v1',account='hm.teoh.2024',seed=9217,
        arms=['restored_context_control','selective_abstention'],refusal=REFUSAL,
        training_slots_per_epoch=1120,unique_inputs=dict(restored_context_control=1000,selective_abstention=1120),
        ordinary=790,unanswerable=120,supported_counterparts=120,self_contained_controls=90,
        unanswerable_fraction=120/1120,unanswerable_kinds=dict(empty=40,irrelevant=40,adversarial=40),
        validation=dict(cases=len(validation),missing=18,answerable=30),challenge=dict(cases=len(challenge),missing=24,answerable=36),
        optimizer_steps=280,epochs=2,effective_batch=8,learning_rate=1e-5,weight_decay=.01,
        snapshots=[0,70,140,280],
        matching='Same initial English LoRA, slot shuffles, optimizer/dropout seeds,8 example-mean reply CEs per update. Control replaces each missing-context slot with its supported counterpart, never assigns an invented fact to an unsupported input. Different text lengths imply different token-level dropout streams.',
        selection='Generated48-case validation only: maximize exact-standard-response answerability balanced score, require canonical over-refusal<=0.10, supported-value matches>=baseline-1 and no generation failures; step0 eligible. Teacher CE diagnostic only. Challenge/user outputs cannot select checkpoints.',
        source_english_sha256='55a7f02626c08894bfdc4b44f68dacf86e8f4edafb7aa6bd78aead9162698913',
        frozen_singlish_sha256='6ea7066deda4f4e9e5f4a9e070d46427536a0dac05d5a416864e97c2dc3f5db2',
        source_base_parameters=3178255360,trainable_parameters=10485760,
        inference=dict(num_beams=4,no_repeat_ngram_size=3,max_new_tokens=64,total_limit=384),
        policy='The English model learns the standard refusal. Only an emitted exact refusal bypasses Singlish rewriting; no input-pattern refusal rules or context-policy layer are used in this experiment.',
        provenance='New pairs and controls are assistant-authored synthetic data, frozen before baseline or training. Different validation/challenge scenarios and pressure wording; not independent human gold or project locked test.',
        default_changed=False,source_weights_preserved=True,
        research_source='https://arxiv.org/abs/1806.03822',
        data_sha256={p.name:sha(p) for p in sorted((OUT/'data').glob('*.json'))})
    save('recipe.json',recipe)
    print(json.dumps(dict(counts={name:len(rows) for name,rows in [('train',train),('control',control),('validation',validation),('challenge',challenge)]},refusal_fraction=recipe['unanswerable_fraction'],audit=audit)))


if __name__=='__main__':main()
