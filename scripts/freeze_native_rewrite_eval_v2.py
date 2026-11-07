"""Unlabeled fresh student conversations, frozen before the three training arms."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'data/public/native_rewrite_v2'
DEV=[
([('peer','I finally handed in the lab report.'),('user','You were working on that all night.')],'I can finally sleep without thinking about the deadline.'),
([('peer','My eye appointment is tomorrow morning.'),('user','What happened to your eye?')],'There is a painful bump and I want the doctor to check it.'),
([('peer','Want something from the bakery?'),('user','I have not had breakfast yet.')],'Which bread should I get for you?'),
([('peer','I cannot find your usual study spot.'),('user','I am at the table beside the vending machines.')],'Are you near the coffee machine or the snack machine?'),
([('peer','I have the notes you lent me.'),('user','I need them for revision tomorrow.')],'Can I bring your notes back first thing tomorrow?'),
([('peer','Can you proofread my conclusion?'),('user','Yes, send me the draft.')],'I sent it just now, could you check the last two sentences?'),
([('peer','Are you coming for volleyball?'),('user','I cannot play today because I have another appointment.')],'Can you join our game this afternoon after all?'),
([('peer','I missed our project meeting.'),('user','We mostly discussed the survey questions.')],'Could you fill me in on what I need to do next?'),
([('peer','The internship interview is over.'),('user','How did it go?')],'I think it went quite well but I am still overthinking my answers.'),
([('peer','I might take a break from studying.'),('user','Same, I have been sitting here for hours.')],'Do you want to walk around campus for a while?'),
([('peer','Have you started making the poster?'),('user','Not yet, I am collecting the photos first.')],'Have you finished the poster already?'),
([('user','I will wait by the blue sign.'),('user','Take your time getting off the train.')],'I am out of the station now, which sign did you say?'),
([('peer','I borrowed your laptop stand.'),('user','You can use it during your presentation.')],'Where should I leave your stand when I finish?'),
([('peer','This group assignment is stressing me out.'),('user','Which part is giving you trouble?')],'I feel like I am doing most of it on my own.'),
([('peer','I am buying drinks for everyone.'),('user','That is nice of you.')],'Would you prefer a hot drink or an iced one?'),
([('peer','Could we have our rehearsal tonight?'),('user','I am busy tonight, but tomorrow afternoon works.')],'What about rehearsing tonight around eight?'),
([('peer','My tutorial got moved to another building.'),('user','Do you know the new room number?')],'Not yet, I am trying to find the announcement.'),
([('peer','Thank you for checking my slides.'),('user','No problem, your examples were clear.')],'Your comments helped me shorten the presentation a lot.'),
([('peer','I left my card holder with you.'),('user','Yes, it is in my tote bag.')],'Can you keep my card holder safe until class ends?'),
([('peer','What time are we getting lunch?'),('user','I am free once my seminar ends at one.')],'Should we eat at half past one then?'),
]
TEST=[
([('peer','The scholarship results came out.'),('user','You were waiting so long for them.')],'I got it! I still cannot believe they picked me.'),
([('peer','I am heading to a clinic after class.'),('user','Is everything okay?')],'I have not been feeling well, so I booked a check-up.'),
([('peer','Dinner at the new food hall?'),('user','I am pretty hungry already.')],'Do you have a stall you want to try there?'),
([('peer','I am at the foyer waiting for you.'),('user','I am still walking across the car park.')],'Would it be easier if I waited outside instead?'),
([('peer','You left your jacket in my locker.'),('user','Oh, I will need it for tomorrow.')],'Shall I bring your jacket along when we meet?'),
([('peer','I wrote an outline for our report.'),('user','I can look at it after my lecture.')],'Could you check whether the sections are in a sensible order?'),
([('peer','We need one more person for the game.'),('user','I cannot join until I finish work at seven.')],'Can you be at the court by six?'),
([('peer','I forgot to upload the attendance form.'),('user','The portal is still open for another hour.')],'Thanks for reminding me, I will do it now.'),
([('peer','I did badly on the practice quiz.'),('user','Was it much harder than you expected?')],'Yeah, now I am worried about the actual exam.'),
([('peer','I am going to get a late snack.'),('user','I could use a break from this essay.')],'Want to join me for something small to eat?'),
([('peer','Have you booked our discussion room?'),('user','Yes, the booking confirmation is in my email.')],'Remember to book the room before the slots run out.'),
([('user','I got a table beside the glass doors.'),('user','I put my green bottle on it so you can spot me.')],'I just entered the canteen, how do I find your table?'),
([('peer','I have your calculator with me.'),('user','I lent it to you for the tutorial.')],'Can I pass your calculator back after the next class?'),
([('peer','I had a disagreement with my teammate.'),('user','That sounds frustrating.')],'I do not know how to talk to them without making it worse.'),
([('peer','I can help collect the takeaway orders.'),('user','Thanks, that saves us a trip.')],'What would you like me to pick up for you?'),
([('peer','Can we go through the project on Friday?'),('user','I am away on Friday. Monday evening is free.')],'Would Friday afternoon be okay instead?'),
([('peer','Our usual printing room is closed.'),('user','I have not checked the other building yet.')],'Do you know if the printer over there is working?'),
([('peer','The club event is finally over.'),('user','You put in a lot of work organizing it.')],'Thanks for helping out, I could not have managed alone.'),
([('peer','I think my water bottle is in your bag.'),('user','Yes, I picked it up after practice.')],'Could you bring my bottle to practice tomorrow?'),
([('peer','Want to review our notes after lunch?'),('user','I need to leave campus by half past four.')],'Can we start at two and finish by three?'),
]
if (folder/'fresh_eval.json').exists():raise FileExistsError('Preserve frozen new conversations')
rows=[]
for split,group in [('development',DEV),('test',TEST)]:
 for n,(turns,message) in enumerate(group,1):rows.append(dict(item_id=f'NATIVE2-{split}-{n:03}',split=split,group_id=f'fresh-{split}-{n:03}',context=[t for _,t in turns],speakers=[s for s,_ in turns],message=message,response=''))
(folder/'fresh_eval.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
(folder/'fresh_eval_manifest.json').write_text(json.dumps(dict(cases=40,development=20,test=20,targets_supplied=False,source='Assistant-authored new conversational instances; no human gold or pretrained-data novelty claim',model_selection='Only native rewrite validation loss selects checkpoints; these generation cases never enter gradients. All three predeclared test outputs may be compared once after selection.',input_sha256=hashlib.sha256((folder/'fresh_eval.json').read_bytes()).hexdigest()),indent=2)+'\n')
print('Frozen40 fresh conversation instances; no reference replies.')
