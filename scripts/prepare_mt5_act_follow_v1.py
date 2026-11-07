"""Author and freeze development-only paired reply-act supervision.

Each tuple contains one conversation and two context-compatible response goals.
All text is assistant-authored experimental material, not human-reviewed gold.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/training_expansion/mt5_act_follow_v1'
# user turn, peer turn, latest peer message, [(acts, English reply), ...]
TRAIN=[
 ('I am free after the lecture.','We could go out together.','Want to have dinner at the food court?', [('accept', 'Sure, dinner at the food court sounds good.'),('decline','I will pass on dinner, thanks.')]),
 ('I have no plans on Sunday.','A few of us are going cycling.','Would you like to join us?', [('accept','Yes, I would like to join you.'),('decline','Thanks for asking, but I will sit this one out.')]),
 ('I finished my assignment early.','There is a film showing tonight.','Shall we watch it together?', [('accept','Sure, let us watch it tonight.'),('decline','I would rather skip the film tonight, thanks.')]),
 ('I will be on campus tomorrow.','We are playing table tennis after lunch.','Do you want to play?', [('accept','Yes, I would like to play after lunch.'),('decline','I will pass on table tennis tomorrow, thanks.')]),
 ('I have a free hour this afternoon.','There is an art exhibition nearby.','Want to take a look?', [('accept','Sure, let us have a look at the exhibition.'),('decline','Thanks, but I do not feel like visiting the exhibition.')]),
 ('I am taking a break now.','I am heading to the canteen.','Shall I get you a snack?', [('accept','Yes please, a snack would be nice.'),('decline','No thanks, I do not need a snack.')]),
 ('My lab ends at six tonight.','We can work on the slides afterwards.','Can we start at five thirty?', [('decline','I cannot start at five thirty because my lab ends at six.'),('decline counterpropose','I cannot start at five thirty. Could we start after six instead?')]),
 ('I am unavailable on Monday, but Tuesday is free.','I will check the meeting schedule.','Can you join the meeting on Monday?', [('decline','I cannot join the meeting on Monday.'),('decline counterpropose','Monday does not work for me. Could we meet on Tuesday instead?')]),
 ('I must leave the studio by three.','The discussion will take an hour.','Shall we begin at two thirty?', [('decline','An hour from two thirty will be too late; I must leave by three.'),('decline counterpropose','I must leave by three. Could we start the hour-long discussion at two instead?')]),
 ('I only have ten dollars for lunch.','We can choose somewhere nearby.','Do you want the twenty-dollar set?', [('decline','No thanks, twenty dollars is over my ten-dollar lunch budget.'),('decline counterpropose','That is over my ten-dollar budget. Shall we find a cheaper lunch?')]),
 ('My ankle is injured and I cannot run.','I hope it gets better soon.','Can you join our run this evening?', [('decline','I cannot join the run because my ankle is injured.'),('decline counterpropose','I cannot run with my ankle like this. Could we do something seated instead?')]),
 ('I can attend the morning session only.','The organiser has two sessions.','Would you take the afternoon slot?', [('decline','I cannot take the afternoon slot; I am only available in the morning.'),('decline counterpropose','The afternoon does not work for me. Could I take the morning slot instead?')]),
 ('You still have my calculator.','Yes, I borrowed it yesterday.','May I return your calculator after the tutorial?', [('accept','Yes, returning my calculator after the tutorial is fine.'),('clarify','What time does your tutorial finish?')]),
 ('You borrowed my camera for your project.','I have kept it safely.','Can I give it back at the station?', [('accept','Yes, you can return my camera at the station.'),('clarify','Which station do you mean?')]),
 ('I left my headphones with you.','I have them in my bag.','Is it okay to return them tomorrow?', [('accept','Yes, tomorrow is fine for returning my headphones.'),('clarify','What time tomorrow could you return them?')]),
 ('I have not checked the workshop schedule.','The timetable might have changed.','Does the workshop start at eleven?', [('express_uncertainty','I am not sure whether it starts at eleven.'),('express_uncertainty offer_or_commit','I am not sure. I will check the workshop schedule.')]),
 ('I do not know if the sports hall is open tonight.','We may need another place.','Does it shut at eight?', [('express_uncertainty','I do not know whether it shuts at eight.'),('express_uncertainty offer_or_commit','I do not know. I will check the sports hall opening hours.')]),
 ('I have not seen the new assignment deadline.','It may have moved.','Is it due on Friday?', [('express_uncertainty','I am not sure whether it is due on Friday.'),('express_uncertainty offer_or_commit','I am not sure. I will check the assignment deadline.')]),
 ('I sent you the revised spreadsheet this morning.','I will look for it.','Have you sent the revision yet?', [('answer_or_inform','Yes, I sent the revised spreadsheet this morning.'),('answer_or_inform request_action','Yes, I sent it this morning. Please check your inbox.')]),
 ('I already reserved our rehearsal room.','Thanks for arranging it.','Have you made the room booking?', [('answer_or_inform','Yes, I have already reserved the rehearsal room.'),('answer_or_inform request_action','Yes, the rehearsal room is reserved. Please check the booking confirmation.')]),
 ('I have saved the lecture audio on my phone.','I missed that lecture.','Could you send me the audio?', [('offer_or_commit','Sure, I will send you the lecture audio.'),('clarify','Which app should I send the audio through?')]),
 ('I have a spare charger in my bag.','My battery is nearly flat.','Could I borrow your spare charger?', [('accept offer_or_commit','Yes, you can borrow it. I will bring it over.'),('clarify','How long would you need the charger?')]),
 ('You said the report was confusing.','The methods section is hard to follow.','Could you look through that section with me?', [('offer_or_commit','Sure, I will go through the methods section with you.'),('clarify','Which part of the methods section is confusing?')]),
 ('I will be near the library after class.','We could revise together.','Meet me there when class ends.', [('accept','Okay, I will meet you after class.'),('clarify','Where exactly should I meet you?')]),
]
VALIDATION=[
 ('I have some free time on Saturday.','There is a campus concert.','Would you like to come to the concert?', [('accept','Yes, I would like to come to the concert.'),('decline','Thanks, but I will skip the concert.')]),
 ('My shift finishes at nine.','We can have supper afterwards.','Can we meet at eight thirty?', [('decline','I cannot meet at eight thirty because my shift finishes at nine.'),('decline counterpropose','I cannot meet at eight thirty. Could we meet after nine?')]),
 ('I need to leave the cafe by noon.','Our planning session takes an hour.','Can we start at eleven thirty?', [('decline','An hour from eleven thirty runs past noon, so that will not work.'),('decline counterpropose','I must leave by noon. Could we start the hour-long session at eleven?')]),
 ('You have my portable speaker.','I packed it safely.','Can I return your speaker on Friday?', [('accept','Yes, Friday is fine for returning my speaker.'),('clarify','What time on Friday would you return it?')]),
 ('I have not checked the museum opening hours.','The weekend schedule might be different.','Does it open at ten?', [('express_uncertainty','I am not sure whether it opens at ten.'),('express_uncertainty offer_or_commit','I am not sure. I will check the museum opening hours.')]),
 ('I already submitted our survey form.','Good, that was the last thing left.','Have you submitted the form?', [('answer_or_inform','Yes, I have already submitted our survey form.'),('answer_or_inform request_action','Yes, I submitted it. Please check the confirmation message.')]),
 ('I have the lab worksheet on my tablet.','I lost my copy.','Could you send me the worksheet?', [('offer_or_commit','Sure, I will send you the lab worksheet.'),('clarify','Where should I send the worksheet?')]),
 ('I can spend twelve dollars on dinner.','There are several places nearby.','Shall we get the thirty-dollar buffet?', [('decline','The thirty-dollar buffet is over my twelve-dollar budget.'),('decline counterpropose','That buffet is over my budget. Could we find dinner for twelve dollars or less?')]),
]
EVALUATION=[
 ('I have a free evening tomorrow.','Our club has a board-game night.','Would you like to join the games?', [('accept','Yes, I would like to join the games.'),('decline','Thanks, but I will skip the games tomorrow.')]),
 ('I have already finished my revision.','There is a pottery session this weekend.','Want to try it together?', [('accept','Sure, let us try the pottery session.'),('decline','I would rather pass on the pottery session, thanks.')]),
 ('I am volunteering until 8:20 tonight.','We could have dinner after that.','Can you meet for dinner at 7:50?', [('decline','I cannot meet at 7:50 because I am volunteering until 8:20.'),('decline counterpropose','I cannot make 7:50. Could we meet after 8:20 instead?')]),
 ('I am busy on Thursday, but Saturday is free.','I will ask the rest of the group.','Can you attend our practice on Thursday?', [('decline','I cannot attend practice on Thursday.'),('decline counterpropose','Thursday does not work for me. Could we practise on Saturday instead?')]),
 ('I have to leave the workshop by 5:10.','The review takes forty minutes.','Shall we start the review at five?', [('decline','A forty-minute review at five runs past my 5:10 departure, so I cannot do that.'),('decline counterpropose','I must leave by 5:10. Could we start the forty-minute review at 4:30?')]),
 ('I only have seven dollars for breakfast.','We can eat near campus.','Would you like the sixteen-dollar breakfast platter?', [('decline','No thanks, the sixteen-dollar platter is over my seven-dollar budget.'),('decline counterpropose','That platter is over my budget. Could we find breakfast for seven dollars or less?')]),
 ('My shoulder is injured and I cannot lift heavy objects.','Take care of it.','Can you help lift this heavy cabinet?', [('decline','I cannot lift the cabinet because my shoulder is injured.'),('decline counterpropose','I cannot lift it with my injured shoulder. Could we ask someone else?')]),
 ('I can only attend the online briefing.','There is also an in-person briefing.','Will you come to the in-person briefing?', [('decline','I cannot attend in person; I can only join online.'),('decline counterpropose','I cannot attend in person. Could I join the online briefing instead?')]),
 ('You borrowed my tripod for filming.','I have finished using it.','Can I return your tripod at the bus interchange?', [('accept','Yes, you can return my tripod at the bus interchange.'),('clarify','Which bus interchange do you mean?')]),
 ('You still have my reference book.','It is safely in my locker.','Would Monday be okay to return your book?', [('accept','Yes, Monday is fine for returning my book.'),('clarify','What time on Monday could you return it?')]),
 ('I have not checked the swimming pool timetable.','The holiday hours might differ.','Does the pool close at six?', [('express_uncertainty','I do not know whether the pool closes at six.'),('express_uncertainty offer_or_commit','I do not know. I will check the swimming pool timetable.')]),
 ('I do not know the submission date for the photography contest.','The organiser posted an update.','Is the deadline next Wednesday?', [('express_uncertainty','I am not sure whether the deadline is next Wednesday.'),('express_uncertainty offer_or_commit','I am not sure. I will check the contest submission date.')]),
 ('I emailed you the signed consent form yesterday.','I will search for it.','Have you emailed the consent form?', [('answer_or_inform','Yes, I emailed you the signed consent form yesterday.'),('answer_or_inform request_action','Yes, I emailed it yesterday. Please check your inbox.')]),
 ('I have already paid the registration fee.','Thanks for sorting that out.','Have you paid the fee yet?', [('answer_or_inform','Yes, I have already paid the registration fee.'),('answer_or_inform request_action','Yes, I paid the fee. Please check the payment receipt.')]),
 ('I saved the interview transcript on my laptop.','I missed the interview.','Could you share the transcript with me?', [('offer_or_commit','Sure, I will send you the interview transcript.'),('clarify','Which address should I send the transcript to?')]),
 ('I will be near the music building after lunch.','We could work on the poster.','Come and meet me there.', [('accept','Okay, I will come over after lunch.'),('clarify','Where exactly should I meet you?')]),
]


def signature(row):
    return json.dumps([row['context'],row['speakers'],row['message']],ensure_ascii=False,separators=(',',':')).lower()


def main():
    from replycoach.generate.reply_act_v1 import act_vector
    FOLDER.mkdir(parents=True,exist_ok=True)
    sources=[];counts={};hashes={}
    for split,examples in [('train',TRAIN),('validation',VALIDATION),('evaluation',EVALUATION)]:
        records=[]
        for n,(user,peer,message,alternatives) in enumerate(examples,1):
            conversation_id=f'ACTFOLLOW-{split}-{n:03d}'
            base=dict(conversation_id=conversation_id,context=[user,peer],speakers=['user','peer'],message=message)
            sig=signature(base)
            if sig in sources:raise ValueError('Repeated conversation across groups/splits')
            sources.append(sig)
            for j,(acts,target) in enumerate(alternatives,1):
                labels=acts.split();act_vector(labels)
                records.append({**base,'item_id':f'{conversation_id}-{j}', 'reply_acts':labels,'response':target,
                    'split':split,'origin':'assistant-authored development supervision; not independently human reviewed'})
        path=FOLDER/f'{split}.json';path.write_text(json.dumps(records,indent=2)+'\n')
        counts[split]={'conversations':len(examples),'targets':len(records)}
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    old=[]
    for split in ['train','validation']:
        for l in (ROOT/f'data/dev/{split}.jsonl').read_text().splitlines():
            r=json.loads(l);old.append(signature(dict(context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'])))
    for r in json.loads((ROOT/'data/evaluation/component_baseline_v1/inputs.json').read_text()):old.append(signature(r))
    if set(sources)&set(old):raise ValueError('Exact overlap with prior development conversations')
    manifest=dict(experiment='mt5_act_follow_v1',counts=counts,source_sha256=hashes,
        provenance='Explicit assistant-authored pairs, two context-compatible reply-act choices per conversation. New English targets only; original LOCAL targets unchanged.',
        scope='Development experiment. Evaluation conversations are frozen before training and first output inspection, but assistant-authored and structurally related to training; not independent human gold or the locked product test.',
        split_policy='Whole conversation groups kept together; no exact context/speaker/message overlap among new splits or with original 190/50 and prior 24 diagnostics. No claim of broad semantic decontamination.',
        classifier_policy='Only original 190 provisional labels supervise classifier. Alternative replies to paired conversations are not treated as uniquely inferable hidden intentions.',
        evaluation_policy='Primary supplied-act controllability on 16 fresh evaluation conversations with two routes each. Classifier-predicted pipeline evaluated separately as any compatible response, not exact preferred-act accuracy. No tuning on these evaluation outputs during this run.')
    (FOLDER/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(counts)

if __name__=='__main__':main()
