"""Fixed English -> natural/chatty Singlish supervision and held-out examples.

These are assistant-authored preferences, not independently human-reviewed gold.
The English anchors are supplied drafts; this module never generates English.
"""

COMMON=[
('dinner','What are you doing tomorrow evening?','Tomorrow evening, I’m having dinner with my classmates.','Tomorrow evening I go makan with my classmates.'),
('breakfast','When should we get breakfast?','Could we have breakfast before class?','Before class go eat breakfast can?'),
('lunch','Where do you want to go for lunch?','Could we go to the vegetarian stall together for lunch?','Lunch we go vegetarian stall together can?'),
('food_first','Are you coming over now?','I’ll get some food first and join you after that.','I go makan first, then join you.'),
('movie_dinner','What should we do after the movie?','Do you want to get dinner after the movie?','After the movie, go eat dinner can?'),
('snacks','Who can bring snacks for everyone?','I’ll bring snacks for everyone.','Snacks for everyone I bring.'),
('rest','Going anywhere tonight?','I’m staying in tonight. I just want to rest.','Tonight I stay in, just want to rest.'),
('tired','Today was really exhausting.','That sounds tiring. Get some rest when you reach home.','Sounds tiring leh. Reach home then rest a bit.'),
('watch','What do you feel like watching?','I haven’t decided what to watch. Maybe a comedy.','Haven’t decide what to watch yet. Comedy maybe?'),
('celebrate','I got the job!','I’m happy for you! Let’s celebrate this weekend.','Happy for you sia! This weekend we celebrate.'),
('photos','Would you like the photos?','Sure, send me the photos when you’re free.','Can, free then send me the photos.'),
('waiting','I’ll come downstairs in a moment.','I’ll wait downstairs. Message me when you’re coming down.','I wait downstairs. Coming down that time message me.'),
('late','We are ready to start the game.','Sorry, I’m running a little late. Please start without me.','Paiseh, I a bit late. You all start first.'),
('seat','Where would you like to sit?','Could you save me a seat near the aisle?','Aisle there save me a seat can?'),
('food_choice','Rice or noodles today?','I’d rather have noodles than rice today.','Today I rather eat noodles than rice.'),
('light_food','What do you want to eat?','I’ll have something light because I’m not very hungry.','I not very hungry, eat something light can already.'),
('seating','Where should we sit?','Do you want to sit outside or stay indoors?','Want sit outside or stay inside?'),
('group','Do the others want to join?','I’ll ask the others and let you know.','I ask the rest first, then update you.'),
('preference','Which place do you prefer?','I’m fine with either place. You can choose.','Either place also can, you choose.'),
('arrival','I’m waiting for you here.','Thanks for waiting. I’m almost there.','Thanks for waiting ah. I reaching soon.'),
]

VALIDATION=[
('food','What are you doing after class?','I’m getting some food with my sister later.','Later I go makan with my sister.'),
('meal','Should we head back now?','Let’s get something to eat before we head back.','We go makan first, then head back.'),
('evening','Any plans tonight?','I’ll stay home and watch a show tonight.','Tonight I stay home watch a show.'),
('rest','Are you heading out again?','It was a long day. I’m going to rest now.','Long day leh, I go rest now.'),
('address','Do you know where to go?','Could you send me the address again?','Address send me again can?'),
('meet','Does that plan work for you?','I’m okay with that plan. What time are we meeting?','That plan can. Meet what time?'),
('decline','Want to go out tonight?','I don’t feel like going out tonight.','Tonight I don’t feel like going out leh.'),
('invite','What should we do tomorrow?','Would you like to join us for lunch tomorrow?','Tomorrow want join us for lunch?'),
('split_tasks','What should each of us buy?','I’ll buy the drinks. You can get the snacks.','Drinks I buy, snacks you get.'),
('train','Have you arrived yet?','I’m still on the train. I’ll message you when I arrive.','Still on the train. Reach already then I message you.'),
('walk','The weather looks nice today.','The weather is quite nice. Let’s take a short walk.','Weather quite nice leh. Go walk a bit?'),
('cafe','Which café do you prefer?','I think I’ll choose the quieter café.','I think I choose the quieter café.'),
('hungry','Have you eaten?','I haven’t eaten yet. I’ll get food first.','Haven’t eat yet, I go makan first.'),
('patient','Sorry, I’m taking a while.','There’s no rush. Take your time.','No rush lah, take your time.'),
('thanks','I’ve helped with the slides.','Thanks for helping me with the slides.','Thanks for helping me with the slides.'),
('coffee','I might be free later.','If you’re free later, we can get coffee.','Later you free then we go get coffee.'),
('confirm','Can you let me know if you’re joining?','I’m not sure yet. Can I confirm with you tonight?','Not sure yet leh. Tonight confirm with you can?'),
('try','Want to try this game?','That looks fun. I’d like to try it.','Looks fun leh, I want to try.'),
('book','How can I get the book from you?','I’ll leave the book at the counter for you.','Book I leave at the counter for you.'),
('happy','I enjoyed the event.','I’m glad you enjoyed it.','Glad you enjoyed it!'),
]

TEST=[
('dinner','Where should we eat?','Let’s get dinner somewhere nearby.','Nearby find somewhere eat dinner?'),
('choose','Which place should we choose?','Could you pick a place? I’m fine with anything.','You choose the place can? Anything I also okay.'),
('join','Coming to the library now?','I’ll eat first, then meet you at the library.','I go makan first, then meet you at the library.'),
('games','What are your plans for tonight?','I’m going to stay in and play games tonight.','Tonight I stay in play games.'),
('break','I’ve been working all day.','You’ve worked hard today. Take a break.','Today you work hard already, go take a break.'),
('news','I passed my driving test!','That’s great news! I’m happy for you.','Good news sia! Happy for you.'),
('umbrella','Do you think it will rain?','I’ll bring an umbrella just in case.','Umbrella I bring, just in case.'),
('share','Do you want fries?','Do you want to share a plate of fries?','Want share one plate of fries?'),
('later','Is this time okay?','Could we meet a little later?','Meet a bit later can?'),
('company','Want to join me for a meal?','I’m not hungry yet, but I can join you.','I not hungry yet, but can join you.'),
('decide','What should we do next?','Let’s decide after we finish the game.','Game finish then decide.'),
('cafe','What about this café?','That café sounds good. Shall we try it?','That café sounds good leh, try anot?'),
('ready','Are you ready to come down?','I’m still getting ready. I’ll be down soon.','Still getting ready, I coming down soon.'),
('crowd','Where do you want to sit?','I’d prefer to sit somewhere less crowded.','I rather sit somewhere not so crowded.'),
('link','Do you want the link?','Please send the link when you have a moment.','You free then send me the link can?'),
('missed','Did you see my message earlier?','Sorry, I didn’t see your message earlier.','Paiseh, I didn’t see your message earlier.'),
('update','Have you checked yet?','I’ll let you know once I’ve checked.','I check already then update you.'),
('time','Should we hurry?','We can take our time. It’s still early.','Still early, we take our time can.'),
('walk','Would you be okay with walking?','I’m okay with walking if you are.','If you okay to walk, I also okay.'),
('help','I could help you with that.','Thank you! That would really help.','Thanks ah, that would help a lot.'),
]

CONTROLS=[
('lend','Can I borrow your charger?','You can borrow my charger until Friday.','My charger you can borrow until Friday.'),
('return','Who should return the camera?','I need to return the camera you lent me.','The camera you lent me, I need return to you.'),
('keys','When do you need your keys back?','Please return my keys to me tomorrow.','Tomorrow return my keys to me, please.'),
('not_yet','Have you sent the file?','I have not sent the file yet.','File I haven’t send yet.'),
('maybe_late','Will you still come?','I may be late, but I will still come.','I might be late, but I still coming.'),
('money','What would help you?','Could you lend me $23 until Monday?','Lend me $23 until Monday can?'),
('booking','What is the booking?','The booking is for room N416 at 15:20.','Booking is room N416, 15:20.'),
('earliest','What time can you meet?','I cannot meet before 6 pm.','Before 6 pm I cannot meet.'),
('paid','Have you paid Hana?','I’ve already paid $18 to Hana.','$18 I paid Hana already.'),
('jacket','What could you bring?','Do you want me to bring your jacket?','Want me bring your jacket?'),
]


def authored_pairs():
    groups={}
    for split,values in [('common_train',COMMON),('validation',VALIDATION),('test',TEST),('fidelity_controls',CONTROLS)]:
        rows=[]
        for i,(topic,message,english,local) in enumerate(values,1):
            rows.append(dict(item_id=f'REWRITE-{split}-{i:02d}',topic=topic,context=[],speakers=[],message=message,
                standard_anchor=english,response=local,scope='fresh_assistant_'+split,
                anchor_provenance='Assistant-authored fixed English draft for isolated rewriting, not English model output.',
                target_provenance='Assistant-authored natural/chatty style preference, not independent human gold.'))
        groups[split]=rows
    return groups


def schedule(replay_count,overlay_count=120,common_count=20,seed=27149,steps=240):
    import random
    def stream(count,offset):
        cycle=0
        while True:
            values=list(range(count));random.Random(seed+offset+cycle).shuffle(values)
            yield from values;cycle+=1
    old,overlay,common=stream(replay_count,0),stream(overlay_count,10000),stream(common_count,20000)
    return [dict(replay=[next(old) for _ in range(5)],common=next(common),overlay=[next(overlay) for _ in range(2)]) for _ in range(steps)]
