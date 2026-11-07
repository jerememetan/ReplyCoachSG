# Anonymized validation reply review — three seeds

All different reply pairs were shown without arm identity. Judgments were recorded before reading each seed’s arm key. Seed 425 was fixed as the primary qualitative comparison before its replies were inspected. Seeds 1 and 2 were reviewed as exploratory follow-ups after the primary result. This is assistant inspection with shared project/development context, not independent human evaluation.

The same 50 conversations are reused for each seed; pooled cells are not independent conversations. Exact-identical replies were counted separately and were not assumed good. Personal details without prior evidence are allowed as selectable drafts. Some language/relevance boundaries remain judgment calls.

## Seed 425

| Outcome | Count |
| --- | --- |
| sft | 11 |
| dpo | 7 |
| tie | 20 |
| both_bad | 11 |
| identical | 51 |

### 425/1: ITEM-A-030 — LOCAL — tie

- Context: peer: How long is our booking? → user: Oh no, our timer went off.
- Incoming: sian, need book somewhere else, our booking ran out
- Supervised-only: okay, we can book somewhere else
- DPO: Okay, we can book somewhere else
- Review: Only capitalization differs; same useful plan.

### 425/2: ITEM-A-033 — STANDARD — tie

- Context: peer: Hey should we grab lunch? → user: sure! Lets head to Koufu
- Incoming: eh the chicken rice stall open today anot?
- Supervised-only: Yes, it opens today.
- DPO: Yes, the chicken rice stall is open today.
- Review: Both provide a plausible stall-opening answer; no clear quality difference.

### 425/3: ITEM-A-035 — STANDARD — both_bad

- Context: peer: I got to go for my appointment later → user: Yeah I should get going too
- Incoming: don't forget to dabao your food later
- Supervised-only: Okay, I'll be back soon.
- DPO: Okay, I'll be back later.
- Review: Neither addresses the food reminder; both substitute an unrelated return-time update.

### 425/4: ITEM-A-038 — LOCAL — sft

- Context: peer: are we still playing badminton tonight? → user: yeah, I think so
- Incoming: i might be ten mins late leh
- Supervised-only: okay, let me know when you reach
- DPO: okay, i'll be reaching soon
- Review: Responds to the peer’s lateness by requesting an arrival update; A switches to the user’s arrival status.

### 425/5: ITEM-A-044 — STANDARD — both_bad

- Context: peer: have you submitted the project? → user: not yet, I am checking it first
- Incoming: remember ah, dont blur like sotong
- Supervised-only: No worries, I will send it to you later.
- DPO: No worries, I'll send it to you later.
- Review: Both promise sending the project to the peer rather than clearly acknowledging the submission reminder. Near-identical generic sending response.

### 425/6: ITEM-A-044 — LOCAL — dpo

- Context: peer: have you submitted the project? → user: not yet, I am checking it first
- Incoming: remember ah, dont blur like sotong
- Supervised-only: okay, i will send it later
- DPO: okay, I will check it later
- Review: Checking relates to the current project-review task; A gives a less clear sending commitment. Both remain weak reminder responses.

### 425/7: ITEM-A-046 — STANDARD — sft

- Context: peer: want to study together tomorrow? → user: sure, what time?
- Incoming: 2pm u can anot, i can chiong from class
- Supervised-only: okay, 2pm sounds good.
- DPO: Ok, I'll be there at 3pm.
- Review: Directly accepts the proposed 2 pm time. A changes it to 3 pm without framing a counterproposal.

### 425/8: ITEM-B-002 — LOCAL — tie

- Context: peer: I haven't eaten breakfast → user: same sia, I'm hungry also
- Incoming: where you want eat?
- Supervised-only: I'm going to eat at the hawker centre
- DPO: I'm not sure yet.
- Review: A proposes a meal location; B honestly leaves the choice open. Both are valid selectable personal-preference drafts.

### 425/9: ITEM-B-009 — STANDARD — tie

- Context: peer: I'm going library now → user: I'll come over after class
- Incoming: still got space?
- Supervised-only: Yes, I'll be there in a few minutes.
- DPO: Yes, I'm going over now.
- Review: Both start with Yes and add a plausible arrival update; no clear comparative quality difference.

### 425/10: ITEM-B-009 — LOCAL — sft

- Context: peer: I'm going library now → user: I'll come over after class
- Incoming: still got space?
- Supervised-only: Ya, still got space
- DPO: Okay, I'll come over after class
- Review: Directly answers the space-availability question; A only repeats the user’s travel plan.

### 425/11: ITEM-B-029 — LOCAL — tie

- Context: peer: I can't remember which room we're in → user: I'm checking the timetable
- Incoming: which room is our class in?
- Supervised-only: It's in the second floor
- DPO: Our class is in the second floor
- Review: Same floor information with similarly local phrasing; neither supplies a specific room but both offer location information.

### 425/12: ITEM-B-042 — STANDARD — both_bad

- Context: peer: I'm waiting outside already → user: I'm reaching soon
- Incoming: sorry, I took the wrong bus
- Supervised-only: I'm sorry, I didn't take it yet.
- DPO: Okay, I'll take the wrong bus.
- Review: B proposes taking the wrong bus; A is an unclear apology about not taking an unspecified thing. Neither coherently handles the peer’s wrong-bus update.

### 425/13: ITEM-B-048 — STANDARD — tie

- Context: peer: are you coming to the event? → user: I was planning to
- Incoming: the event postpone for thirty minutes
- Supervised-only: okay, I'll be there in about half an hour.
- DPO: Okay, I'll be there in about half an hour.
- Review: Capitalization-only difference.

### 425/14: ITEM-B-048 — LOCAL — tie

- Context: peer: are you coming to the event? → user: I was planning to
- Incoming: the event postpone for thirty minutes
- Supervised-only: okay, I'll be there
- DPO: Okay, I'll be there
- Review: Capitalization-only difference.

### 425/15: ITEM-B-050 — STANDARD — tie

- Context: peer: I haven't exercised in a while → user: same, I've been busy with classes
- Incoming: want to go run later?
- Supervised-only: Sure, I'll run later.
- DPO: Sure, I'll join you later.
- Review: Both accept the run invitation; joining is more explicit but no substantial difference.

### 425/16: ITEM-B-056 — STANDARD — tie

- Context: peer: are you free for badminton tonight? → user: I should be free
- Incoming: the court is already booked for seven
- Supervised-only: Okay, seven works for me.
- DPO: Okay, I'll be there.
- Review: Both naturally confirm the booked badminton plan, with or without repeating seven.

### 425/17: ITEM-B-056 — LOCAL — tie

- Context: peer: are you free for badminton tonight? → user: I should be free
- Incoming: the court is already booked for seven
- Supervised-only: okay, seven works for me
- DPO: Okay, I'll be there
- Review: Both confirm attendance/time for the existing plan.

### 425/18: ITEM-C-001 — STANDARD — tie

- Context: peer: are you going to the library ltr? → user: yeah, probably after dinner
- Incoming: i'll be at lks around 8, want to meet there?
- Supervised-only: Sure, I'll meet you there around 8
- DPO: Sure, I'll meet you there.
- Review: Both accept meeting there; repeating the already stated eight is optional.

### 425/19: ITEM-C-017 — STANDARD — both_bad

- Context: peer: im working on my slides now → user: same, im doing the results section
- Incoming: your part done alr?
- Supervised-only: yes, I am finished it alr.
- DPO: yes, i finished it last week.
- Review: A has broken STANDARD grammar and local alr; B asserts completion last week despite the user just describing work in progress.

### 425/20: ITEM-C-017 — LOCAL — tie

- Context: peer: im working on my slides now → user: same, im doing the results section
- Incoming: your part done alr?
- Supervised-only: yea, i finished it alr
- DPO: ya, i finished it alr
- Review: Yea and ya express the same valid progress answer; no material difference.

### 425/21: ITEM-C-018 — LOCAL — tie

- Context: peer: i'll upload my section before dinner → user: okay, then i can merge everything tonight
- Incoming: eh dont forget your slide ah, we need merge later
- Supervised-only: okay, i'll merge later
- DPO: okay, i shall merge later
- Review: Same planned merging action with a minor modal difference.

### 425/22: ITEM-C-023 — STANDARD — dpo

- Context: peer: saturday im quite free → user: same, no plans yet
- Incoming: want me to jio you for the night market thing?
- Supervised-only: Sure, let me know when you're free.
- DPO: Sure, I'll be there.
- Review: Accepts the invitation; A asks when the peer is free despite Saturday availability already being given.

### 425/23: ITEM-C-038 — STANDARD — tie

- Context: peer: ive finished drafting the conclusion → user: great
- Incoming: can help me check if this part sounds too long?
- Supervised-only: Sure, let me check it for you.
- DPO: Sure, I'll check it for you.
- Review: Equivalent offers to check the requested passage.

### 425/24: ITEM-C-038 — LOCAL — sft

- Context: peer: ive finished drafting the conclusion → user: great
- Incoming: can help me check if this part sounds too long?
- Supervised-only: okay, let me check
- DPO: okay, let me check if it works for you.
- Review: Cleanly offers to check; B shifts to the less specific question of whether it works for the peer.

### 425/25: ITEM-C-041 — STANDARD — tie

- Context: peer: could you upload your section before 8? → user: yes, i'll do it
- Incoming: paiseh, i only just finished class. i'll upload a bit later
- Supervised-only: ok, i'll send you later
- DPO: ok, i'll send it later
- Review: Both offer a vaguely specified sending update; no clear preference. The user also has a section to upload, so this is not automatically a role reversal.

### 425/26: ITEM-C-041 — LOCAL — dpo

- Context: peer: could you upload your section before 8? → user: yes, i'll do it
- Incoming: paiseh, i only just finished class. i'll upload a bit later
- Supervised-only: okay, i'll send you later
- DPO: okay, i'll upload it later
- Review: States the upload action with a clear object; B’s send you later is malformed. Own-section upload is established in context.

### 425/27: ITEM-C-045 — LOCAL — sft

- Context: peer: we said we'd sit together tmr → user: yep
- Incoming: you still coming early to get seats right?
- Supervised-only: yep, i reaching early
- DPO: yep, i still waiting for seats
- Review: Confirms early arrival as requested; B ambiguously describes waiting for seats now for a tomorrow plan.

### 425/28: ITEM-C-058 — LOCAL — sft

- Context: peer: im heading to the court now → user: okay, i'll leave soon
- Incoming: court still occupied leh, need wait a bit
- Supervised-only: okay, i'll leave soon
- DPO: okay, i'll wait for you
- Review: Consistent with the user’s stated departure plan; B turns court occupancy into an unclear wait-for-peer action.

### 425/29: ITEM-D-007 — STANDARD — tie

- Context: peer: u wanna try the matcha at luckin? → user: nahh
- Incoming: then what u wanna drink later?
- Supervised-only: I'm not sure.
- DPO: I'm not sure yet.
- Review: Equivalent uncertainty about a drink preference.

### 425/30: ITEM-D-007 — LOCAL — sft

- Context: peer: u wanna try the matcha at luckin? → user: nahh
- Incoming: then what u wanna drink later?
- Supervised-only: idk leh, i think i'll join later
- DPO: okay, i'll join later
- Review: Idk at least answers the drink-choice question; A merely says joining later. The extra joining detail is still weak.

### 425/31: ITEM-D-009 — STANDARD — dpo

- Context: peer: eh i struggling to do the linear algebra qn → user: which
- Incoming: neh the gaussian elim qn
- Supervised-only: eh, i think i'll do it myself
- DPO: eh, i think i'll do it for you
- Review: Offers to help with the peer’s problem; doing it myself misses who is asking for help.

### 425/32: ITEM-D-009 — LOCAL — tie

- Context: peer: eh i struggling to do the linear algebra qn → user: which
- Incoming: neh the gaussian elim qn
- Supervised-only: eh i dont think i can do it
- DPO: eh i think i'll do it later
- Review: Deferring help and declining because of ability are both plausible selectable replies; no established personal availability/ability decides between them.

### 425/33: ITEM-D-029 — STANDARD — sft

- Context: peer: eh so how, you still want bid for the morning class? → user: yah can
- Incoming: ok then rmb to bid before bidding ends
- Supervised-only: okay, thanks.
- DPO: Okay, I'll check it for you.
- Review: Acknowledges the reminder about the user’s own bidding; A switches to checking something for the peer.

### 425/34: ITEM-D-029 — LOCAL — both_bad

- Context: peer: eh so how, you still want bid for the morning class? → user: yah can
- Incoming: ok then rmb to bid before bidding ends
- Supervised-only: okay, i will send u the link
- DPO: okay, i will send you the link
- Review: Both send a link rather than respond meaningfully to the user’s class-bidding reminder. Only spelling differs.

### 425/35: ITEM-D-033 — STANDARD — both_bad

- Context: peer: hello → user: hi → peer: i am already at the bus stop
- Incoming: wya
- Supervised-only: okay, i will take a bus to get there
- DPO: okay, i will take you there
- Review: Neither clearly answers where the user is; A offers taking the peer somewhere and B offers a bus journey.

### 425/36: ITEM-D-033 — LOCAL — dpo

- Context: peer: hello → user: hi → peer: i am already at the bus stop
- Incoming: wya
- Supervised-only: okay, i am reaching soon
- DPO: okay, i reached the bus stop
- Review: Provides a specific plausible user location; B only gives an arrival estimate to a where question.

### 425/37: ITEM-D-037 — STANDARD — sft

- Context: user: have you reached? → user: wya
- Incoming: i reached, i waiting outside mac
- Supervised-only: okay, i'll wait for you.
- DPO: okay, i'll take you there
- Review: Acknowledges the peer’s update and gives a plausible waiting plan; A offers to take someone to an unspecified place when no transport was requested.

### 425/38: ITEM-D-038 — LOCAL — both_bad

- Context: peer: eh wya → user: i am outside mac → user: u eh
- Incoming: I am reaching soon, wait for me before going in.
- Supervised-only: ok, i am reaching soon
- DPO: ok, i reaching soon
- Review: User already stated they are outside Mac; both suggestions say the user is arriving soon rather than waiting for the arriving peer.

### 425/39: ITEM-D-050 — LOCAL — dpo

- Context: user: eh u ok? why u never join us for bb session yesterday.
- Incoming: woah sorry, i came up with a flu.
- Supervised-only: sorry, i dont think i will be able to join us for bb session
- DPO: okay, let me know when u can come.
- Review: Responds to the ill peer about future participation; A apologizes as if the user missed the session and muddles us.

### 425/40: ITEM-E-007 — STANDARD — both_bad

- Context: peer: tomorrow what you want to eat → user: lets eat at Chinatown tomorrow
- Incoming: onz, tmr meet na li
- Supervised-only: okay, tmr i meet u
- DPO: okay, i'll meet you tomorrow.
- Review: Peer asks where to meet; both merely repeat tomorrow without giving a location. A also uses shorthand in STANDARD.

### 425/41: ITEM-E-007 — LOCAL — both_bad

- Context: peer: tomorrow what you want to eat → user: lets eat at Chinatown tomorrow
- Incoming: onz, tmr meet na li
- Supervised-only: okay, tmr i meet u
- DPO: okay, i meet u tmr
- Review: Both omit the requested meeting location; same sentence with reordered tomorrow.

### 425/42: ITEM-E-015 — LOCAL — tie

- Context: peer: Hi guys, remember to do the quiz, deadline is next week → user: yep noted
- Incoming: hi have you done the quiz alr anot
- Supervised-only: yep, i finished it alr
- DPO: ya, i finished it alr
- Review: Equivalent selectable quiz-completion status.

### 425/43: ITEM-E-025 — STANDARD — sft

- Context: peer: hi class, I had a family emergency and there will not be any class tomorrow. → user: oh no! what about the in class quiz?
- Incoming: will everyone be free on thursday instead for a make up class instead? paiseh
- Supervised-only: yeah, thursday is fine.
- DPO: Sorry, I can't make up for thursday.
- Review: Fluent acceptance; A’s cannot make up for Thursday is awkward for declining attendance. Preference is wording, not a rule that accepting is better than declining.

### 425/44: ITEM-E-028 — LOCAL — both_bad

- Context: user: bro you want to eat at kopitiam later? The whole gang is cmg → user: if you want ah let us know
- Incoming: shag I damm busy sia, i cmi leh, dui bu qi
- Supervised-only: okay, i can meet you at kopitiam later
- DPO: okay, i shall meet you at kopitiam later
- Review: Peer explicitly cannot attend; both still say the user will meet that peer at the gathering.

### 425/45: ITEM-E-032 — STANDARD — sft

- Context: peer: I reminded the class the tutorial is at SR3-1 liao → user: can you change today's tutorial to be in SR3-2 instead?
- Incoming: sorry ah but that one cannot, someone book liao
- Supervised-only: okay, let me know when the class starts
- DPO: okay, I'll book it for you.
- Review: Acknowledges the unavailable room and asks a relevant class-time follow-up; B promises booking the unavailable option without clarifying an alternative.

### 425/46: ITEM-E-032 — LOCAL — both_bad

- Context: peer: I reminded the class the tutorial is at SR3-1 liao → user: can you change today's tutorial to be in SR3-2 instead?
- Incoming: sorry ah but that one cannot, someone book liao
- Supervised-only: okay, i booked for you
- DPO: okay, i booked alr
- Review: Both imply completing a booking after the requested room was stated unavailable; no alternative room is specified.

### 425/47: ITEM-E-040 — STANDARD — tie

- Context: peer: so much work this week wtf → user: ya sia
- Incoming: how ah, want meet together to do work?
- Supervised-only: yeah, I'll meet you there.
- DPO: Sure, I'll meet you there.
- Review: Both naturally accept meeting to work; minor wording only.

### 425/48: ITEM-E-040 — LOCAL — tie

- Context: peer: so much work this week wtf → user: ya sia
- Incoming: how ah, want meet together to do work?
- Supervised-only: can, i meet y'all after work
- DPO: can, i meet you another day
- Review: Both offer plausible new meeting times; no availability evidence establishes a preference.

### 425/49: ITEM-E-046 — LOCAL — dpo

- Context: peer: hello,what time will you release the lecture recordings? → user: usually around 10pm its out
- Incoming: ni hao, can I releasae it at 5 instead?
- Supervised-only: can, i leave it at 5
- DPO: can, 5 works for me
- Review: Clearly accepts five; A uses the wrong/unclear leave action for releasing recordings.

Identical outputs for this seed are preserved in the ignored blinded packet and raw run reports; equality is not an acceptability rating.

## Seed 1

| Outcome | Count |
| --- | --- |
| sft | 8 |
| dpo | 10 |
| tie | 18 |
| both_bad | 15 |
| identical | 49 |

### 1/1: ITEM-A-010 — LOCAL — both_bad

- Context: peer: this course is so tough. → user: yeah, I could really use some help too.
- Incoming: peer: we very kiasu sia, first lesson already like that
- Supervised-only: okay, i shall try to help u.
- DPO: okay, I'll take it.
- Review: Neither clearly responds to the peer’s remark about shared anxiety; take it is unrelated and a generic help offer misses the remark.

### 1/2: ITEM-A-024 — STANDARD — sft

- Context: peer: Hey have you printed the worksheet given out yesterday? → user: No, I will need to print it in school. Do you know how to use the printer?
- Incoming: paiseh, i don't know how to use the printer in school.
- Supervised-only: Okay, I'll print it for you.
- DPO: Okay, let me know how to use the printer.
- Review: A asks again for instructions from a peer who just said they do not know. B is a plausible prospective printing offer; no explicit user inability was established.

### 1/3: ITEM-A-024 — LOCAL — sft

- Context: peer: Hey have you printed the worksheet given out yesterday? → user: No, I will need to print it in school. Do you know how to use the printer?
- Incoming: paiseh, i don't know how to use the printer in school.
- Supervised-only: Okay, I will print it later.
- DPO: Okay, let me know how to use the printer.
- Review: Own printing later is consistent with the user’s stated need; B repeats the unanswered instruction request to a peer who does not know.

### 1/4: ITEM-A-030 — STANDARD — tie

- Context: peer: How long is our booking? → user: Oh no, our timer went off.
- Incoming: sian, need book somewhere else, our booking ran out
- Supervised-only: Yes, we can book a place near the hotel.
- DPO: No worries, I'll book somewhere else.
- Review: Both propose a new booking; an unmentioned hotel is allowed as a selectable new detail. No strong preference.

### 1/5: ITEM-A-030 — LOCAL — tie

- Context: peer: How long is our booking? → user: Oh no, our timer went off.
- Incoming: sian, need book somewhere else, our booking ran out
- Supervised-only: Okay, we can book somewhere else
- DPO: okay, we can book somewhere else
- Review: Capitalization-only difference.

### 1/6: ITEM-A-033 — STANDARD — tie

- Context: peer: Hey should we grab lunch? → user: sure! Lets head to Koufu
- Incoming: eh the chicken rice stall open today anot?
- Supervised-only: Yes, it opens today.
- DPO: Yes, the chicken rice stall is open today.
- Review: Both offer a plausible opening-status answer.

### 1/7: ITEM-A-033 — LOCAL — dpo

- Context: peer: Hey should we grab lunch? → user: sure! Lets head to Koufu
- Incoming: eh the chicken rice stall open today anot?
- Supervised-only: eh, i might be able to meet you at koufu
- DPO: Yep, it opens today
- Review: Directly answers stall opening; B switches to the already established meal-meeting plan.

### 1/8: ITEM-A-035 — LOCAL — both_bad

- Context: peer: I got to go for my appointment later → user: Yeah I should get going too
- Incoming: don't forget to dabao your food later
- Supervised-only: Okay, I'll meet you later
- DPO: Okay, I'll be back soon
- Review: Neither meaningfully acknowledges taking food away; both give an unrelated personal meeting/return update.

### 1/9: ITEM-A-038 — STANDARD — tie

- Context: peer: are we still playing badminton tonight? → user: yeah, I think so
- Incoming: i might be ten mins late leh
- Supervised-only: Okay, I'll be there in ten minutes
- DPO: Okay, I'll be there in about half an hour
- Review: Both provide possible independent user arrival times; unknown personal timing is allowed.

### 1/10: ITEM-A-038 — LOCAL — tie

- Context: peer: are we still playing badminton tonight? → user: yeah, I think so
- Incoming: i might be ten mins late leh
- Supervised-only: Okay, I'll be reaching around nine
- DPO: okay, I'll be reaching soon
- Review: Both provide possible arrival updates; no established time rules out nine.

### 1/11: ITEM-A-044 — STANDARD — both_bad

- Context: peer: have you submitted the project? → user: not yet, I am checking it first
- Incoming: remember ah, dont blur like sotong
- Supervised-only: sorry, I am blurred. I will send it to you later.
- DPO: Sorry, I am blurred. I will check it later.
- Review: Both use the unnatural STANDARD phrase I am blurred and miss the submission reminder.

### 1/12: ITEM-A-046 — STANDARD — tie

- Context: peer: want to study together tomorrow? → user: sure, what time?
- Incoming: 2pm u can anot, i can chiong from class
- Supervised-only: Sure, I'll be there at 3pm.
- DPO: Sure, I'll be there by 3pm.
- Review: Both give an unframed three-o’clock counterproposal rather than directly accepting two; no clear comparative difference.

### 1/13: ITEM-A-054 — LOCAL — tie

- Context: peer: We are going to the club later right? → user: Yeah, I will go back to shower first.
- Incoming: Aiyo, don't need la, let's go now
- Supervised-only: Okay, let's go now
- DPO: Ok, let's go now
- Review: Equivalent acceptance of going now.

### 1/14: ITEM-B-042 — STANDARD — both_bad

- Context: peer: I'm waiting outside already → user: I'm reaching soon
- Incoming: sorry, I took the wrong bus
- Supervised-only: I'm sorry, I took the wrong bus.
- DPO: I'm sorry, I didn't catch it.
- Review: A copies the peer’s wrong-bus status into the user’s reply; B is an unclear unrelated transport apology.

### 1/15: ITEM-B-043 — STANDARD — tie

- Context: peer: a few of us are going out this weekend → user: sounds fun
- Incoming: you want come?
- Supervised-only: yes, I'd like to come.
- DPO: Sure, I'll be there.
- Review: Both naturally accept the invitation.

### 1/16: ITEM-B-048 — LOCAL — dpo

- Context: peer: are you coming to the event? → user: I was planning to
- Incoming: the event postpone for thirty minutes
- Supervised-only: Okay, I'll be there in a few minutes
- DPO: Okay, I'll be there in about half an hour
- Review: Aligns the arrival with the stated half-hour postponement; B could arrive early but does not address the new schedule as clearly.

### 1/17: ITEM-B-050 — STANDARD — tie

- Context: peer: I haven't exercised in a while → user: same, I've been busy with classes
- Incoming: want to go run later?
- Supervised-only: Sure, I'll join you later.
- DPO: Sure, I'll run later.
- Review: Both accept running together; slight wording difference.

### 1/18: ITEM-B-050 — LOCAL — tie

- Context: peer: I haven't exercised in a while → user: same, I've been busy with classes
- Incoming: want to go run later?
- Supervised-only: Sure, I'll join later.
- DPO: Sure, I'll run later.
- Review: Both accept the run invitation; no substantive difference.

### 1/19: ITEM-B-056 — LOCAL — sft

- Context: peer: are you free for badminton tonight? → user: I should be free
- Incoming: the court is already booked for seven
- Supervised-only: Okay, I'll be free for seven
- DPO: Okay, I'll be free after 7
- Review: Confirms availability for seven; A changes to after seven without framing the change.

### 1/20: ITEM-C-017 — STANDARD — both_bad

- Context: peer: im working on my slides now → user: same, im doing the results section
- Incoming: your part done alr?
- Supervised-only: yes, i finished it last week.
- DPO: yes, i finished it alr.
- Review: A uses alr in STANDARD; B places completion last week while user describes working on the section now.

### 1/21: ITEM-C-018 — STANDARD — dpo

- Context: peer: i'll upload my section before dinner → user: okay, then i can merge everything tonight
- Incoming: eh dont forget your slide ah, we need merge later
- Supervised-only: okay, i'll forget it later
- DPO: okay, i'll merge later
- Review: Merge preserves the planned task; forget explicitly conflicts with the reminder.

### 1/22: ITEM-C-018 — LOCAL — dpo

- Context: peer: i'll upload my section before dinner → user: okay, then i can merge everything tonight
- Incoming: eh dont forget your slide ah, we need merge later
- Supervised-only: okay, i shall forget it later
- DPO: okay, i'll merge later
- Review: Merge preserves the planned task; forget conflicts with the reminder.

### 1/23: ITEM-C-023 — STANDARD — dpo

- Context: peer: saturday im quite free → user: same, no plans yet
- Incoming: want me to jio you for the night market thing?
- Supervised-only: Sure, let me know when you're free.
- DPO: Sure, I'll join you for the night market.
- Review: Directly accepts the specific night-market invitation; B redundantly asks for availability already supplied.

### 1/24: ITEM-C-028 — STANDARD — sft

- Context: peer: im waiting at the cafe alr → user: coming now
- Incoming: paiseh, i ordered first because i was too hungry
- Supervised-only: okay, i'll take a look at the menu
- DPO: okay, i'll order for you later.
- Review: Looking at the menu fits joining a peer who already ordered; ordering for that peer ignores the stated progress.

### 1/25: ITEM-C-038 — LOCAL — tie

- Context: peer: ive finished drafting the conclusion → user: great
- Incoming: can help me check if this part sounds too long?
- Supervised-only: okay, I'll check it for you
- DPO: okay, let me check it for a while
- Review: Both offer the requested check; minor phrasing difference.

### 1/26: ITEM-C-041 — STANDARD — tie

- Context: peer: could you upload your section before 8? → user: yes, i'll do it
- Incoming: paiseh, i only just finished class. i'll upload a bit later
- Supervised-only: ok, i'll send it later
- DPO: ok, i'll upload it later
- Review: Both give an own-section sending/uploading update; no clear relative distinction.

### 1/27: ITEM-C-045 — LOCAL — sft

- Context: peer: we said we'd sit together tmr → user: yep
- Incoming: you still coming early to get seats right?
- Supervised-only: yep, i reaching early
- DPO: yep, i reaching early already
- Review: More natural early-arrival confirmation; adding already to a future plan is awkward.

### 1/28: ITEM-D-009 — STANDARD — dpo

- Context: peer: eh i struggling to do the linear algebra qn → user: which
- Incoming: neh the gaussian elim qn
- Supervised-only: oh yeah, i thought i was thinking of doing the algebra qn
- DPO: eh, i think i'll do it for you
- Review: Coherent offer to help the peer; A is convoluted and does not address the clarified question.

### 1/29: ITEM-D-010 — LOCAL — tie

- Context: peer: hello, can i ask u some question idk how to do → user: uhh i see if i can help
- Incoming: uhh, u know the guassian elim qn
- Supervised-only: ya i know
- DPO: yeah i know
- Review: Equivalent knowledge confirmation.

### 1/30: ITEM-D-029 — STANDARD — both_bad

- Context: peer: eh so how, you still want bid for the morning class? → user: yah can
- Incoming: ok then rmb to bid before bidding ends
- Supervised-only: Okay, I'll send it to you later.
- DPO: Okay, I'll see you later.
- Review: Both replace the bidding reminder with an unrelated sending/meeting commitment.

### 1/31: ITEM-D-029 — LOCAL — both_bad

- Context: peer: eh so how, you still want bid for the morning class? → user: yah can
- Incoming: ok then rmb to bid before bidding ends
- Supervised-only: okay, i will send you a link
- DPO: okay, i will send you the link
- Review: Both promise a link rather than respond to the bidding reminder; article change does not fix relevance.

### 1/32: ITEM-D-033 — STANDARD — both_bad

- Context: peer: hello → user: hi → peer: i am already at the bus stop
- Incoming: wya
- Supervised-only: okay, i will take a bus to get there
- DPO: okay, i will take you there
- Review: Neither provides a clear location in response to where are you.

### 1/33: ITEM-D-033 — LOCAL — sft

- Context: peer: hello → user: hi → peer: i am already at the bus stop
- Incoming: wya
- Supervised-only: okay, i reached the bus stop
- DPO: okay, i am reaching soon
- Review: Specific plausible bus-stop location answers where; A only gives an arrival estimate.

### 1/34: ITEM-D-037 — LOCAL — tie

- Context: user: have you reached? → user: wya
- Incoming: i reached, i waiting outside mac
- Supervised-only: okay, i reaching soon
- DPO: okay, i reached alr
- Review: Arrived and arriving soon are both possible personal statuses with no earlier user location constraint.

### 1/35: ITEM-D-038 — LOCAL — dpo

- Context: peer: eh wya → user: i am outside mac → user: u eh
- Incoming: I am reaching soon, wait for me before going in.
- Supervised-only: ok, i reaching soon
- DPO: ok, i will wait for you
- Review: User already outside Mac should wait for the peer; A says user is arriving instead.

### 1/36: ITEM-D-046 — STANDARD — both_bad

- Context: peer: eh help → user: what
- Incoming: the cinema ticket is sold out bruh
- Supervised-only: okay, I'll take it.
- DPO: okay, I'll take it for you
- Review: Both offer to take an unavailable ticket without a coherent response to it being sold out.

### 1/37: ITEM-D-046 — LOCAL — both_bad

- Context: peer: eh help → user: what
- Incoming: the cinema ticket is sold out bruh
- Supervised-only: okay, i'll send it to you later
- DPO: okay, i'll take it for you
- Review: Neither clearly handles sold-out tickets; both give generic acquisition/sending actions without a clear alternative.

### 1/38: ITEM-D-050 — STANDARD — tie

- Context: user: eh u ok? why u never join us for bb session yesterday.
- Incoming: woah sorry, i came up with a flu.
- Supervised-only: oh, sorry, i didn't see u today.
- DPO: oh, sorry, i didn't see u.
- Review: Equivalent apologies about not seeing the peer; neither handles illness particularly well.

### 1/39: ITEM-D-050 — LOCAL — both_bad

- Context: user: eh u ok? why u never join us for bb session yesterday.
- Incoming: woah sorry, i came up with a flu.
- Supervised-only: sorry, i dont think i will be able to join us for bb session
- DPO: okay, i'll join u later
- Review: A apologizes as if user missed the session; B gives a contextless joining plan instead of addressing the peer’s illness.

### 1/40: ITEM-E-005 — STANDARD — both_bad

- Context: peer: i heard you going to cafe later to study → user: ya you want come? dont say i never jio
- Incoming: can ah, later i join you
- Supervised-only: can, i join you later
- DPO: ok, i join later
- Review: Both are weak STANDARD phrasing and shift to user joining after the peer accepted joining them.

### 1/41: ITEM-E-007 — STANDARD — dpo

- Context: peer: tomorrow what you want to eat → user: lets eat at Chinatown tomorrow
- Incoming: onz, tmr meet na li
- Supervised-only: sorry, i can't make it tomorrow.
- DPO: okay, I'll meet you there
- Review: There refers to the established Chinatown plan. A announces cancellation instead of addressing the location question.

### 1/42: ITEM-E-007 — LOCAL — both_bad

- Context: peer: tomorrow what you want to eat → user: lets eat at Chinatown tomorrow
- Incoming: onz, tmr meet na li
- Supervised-only: okay, tmr i meet u
- DPO: can, i meet u tomorrow
- Review: Neither supplies or confirms the requested meeting location.

### 1/43: ITEM-E-009 — STANDARD — sft

- Context: peer: later I need to lend your charger to charge my phone → user: ok can but I can only give you after 3
- Incoming: hey can confirm i can use your charger after 3? thanks ah
- Supervised-only: Sure, I'll send you after 3
- DPO: Sure, I'll be able to use it after 3
- Review: Despite an omitted object, the intended charger handoff addresses the request; A shifts use of the charger to the user.

### 1/44: ITEM-E-009 — LOCAL — sft

- Context: peer: later I need to lend your charger to charge my phone → user: ok can but I can only give you after 3
- Incoming: hey can confirm i can use your charger after 3? thanks ah
- Supervised-only: okay, I'll send you after 3
- DPO: Okay, I'll be able to use it after 3
- Review: Offers the handoff after three; B shifts who will use the charger.

### 1/45: ITEM-E-025 — LOCAL — tie

- Context: peer: hi class, I had a family emergency and there will not be any class tomorrow. → user: oh no! what about the in class quiz?
- Incoming: will everyone be free on thursday instead for a make up class instead? paiseh
- Supervised-only: yeah, thursday is fine.
- DPO: okay, i'll be free on thursday
- Review: Both naturally confirm Thursday availability.

### 1/46: ITEM-E-027 — LOCAL — dpo

- Context: peer: eh the school got 3d printer anot how to use → user: have, i using it now, you need to book a slot to use → peer: ok i book later to use
- Incoming: paiseh ah can i change my booking i want to use it later
- Supervised-only: can, i book later
- DPO: can, i change it later
- Review: Changing the booking addresses the peer’s request; booking later changes the action.

### 1/47: ITEM-E-028 — LOCAL — both_bad

- Context: user: bro you want to eat at kopitiam later? The whole gang is cmg → user: if you want ah let us know
- Incoming: shag I damm busy sia, i cmi leh, dui bu qi
- Supervised-only: okay, we meet at kopitiam later
- DPO: okay, i shall meet you at kopitiam later
- Review: Both keep a meeting with a peer who explicitly said they cannot attend.

### 1/48: ITEM-E-032 — LOCAL — both_bad

- Context: peer: I reminded the class the tutorial is at SR3-1 liao → user: can you change today's tutorial to be in SR3-2 instead?
- Incoming: sorry ah but that one cannot, someone book liao
- Supervised-only: can, i book for you
- DPO: okay, i booked alr
- Review: Both suggest booking the requested room despite its stated unavailability.

### 1/49: ITEM-E-040 — STANDARD — tie

- Context: peer: so much work this week wtf → user: ya sia
- Incoming: how ah, want meet together to do work?
- Supervised-only: yeah, I'll meet you there.
- DPO: Sure, I'll meet you there.
- Review: Equivalent acceptance of meeting to work.

### 1/50: ITEM-E-040 — LOCAL — tie

- Context: peer: so much work this week wtf → user: ya sia
- Incoming: how ah, want meet together to do work?
- Supervised-only: can, i meet y'all after work
- DPO: can, i meet you guys at the mrt
- Review: Both propose plausible meeting details; no established availability/location makes one better.

### 1/51: ITEM-E-046 — LOCAL — dpo

- Context: peer: hello,what time will you release the lecture recordings? → user: usually around 10pm its out
- Incoming: ni hao, can I releasae it at 5 instead?
- Supervised-only: can, i leave it at 5
- DPO: can, 5 works for me
- Review: Directly accepts the proposed release time; leave it is the wrong/unclear action.

Identical outputs for this seed are preserved in the ignored blinded packet and raw run reports; equality is not an acceptability rating.

## Seed 2

| Outcome | Count |
| --- | --- |
| sft | 13 |
| dpo | 7 |
| tie | 29 |
| both_bad | 10 |
| identical | 41 |

### 2/1: ITEM-A-010 — LOCAL — both_bad

- Context: peer: this course is so tough. → user: yeah, I could really use some help too.
- Incoming: peer: we very kiasu sia, first lesson already like that
- Supervised-only: okay, i will take another one.
- DPO: okay, I'll take it.
- Review: Neither responds to the shared course-anxiety remark; both introduce an unclear take action.

### 2/2: ITEM-A-024 — STANDARD — both_bad

- Context: peer: Hey have you printed the worksheet given out yesterday? → user: No, I will need to print it in school. Do you know how to use the printer?
- Incoming: paiseh, i don't know how to use the printer in school.
- Supervised-only: Okay, let me know how to use the printer.
- DPO: Okay, I'll see you later.
- Review: A asks for instructions from a peer who just said they do not know; B substitutes an unrelated farewell.

### 2/3: ITEM-A-030 — STANDARD — tie

- Context: peer: How long is our booking? → user: Oh no, our timer went off.
- Incoming: sian, need book somewhere else, our booking ran out
- Supervised-only: Yes, we can book somewhere else.
- DPO: Yes, we need to book somewhere else.
- Review: Both acknowledge the need for a new booking.

### 2/4: ITEM-A-033 — LOCAL — tie

- Context: peer: Hey should we grab lunch? → user: sure! Lets head to Koufu
- Incoming: eh the chicken rice stall open today anot?
- Supervised-only: yep, the chicken rice stall opens today
- DPO: Yep, the chicken rice stall open today
- Review: Equivalent plausible stall-opening answers in LOCAL.

### 2/5: ITEM-A-035 — STANDARD — both_bad

- Context: peer: I got to go for my appointment later → user: Yeah I should get going too
- Incoming: don't forget to dabao your food later
- Supervised-only: Okay, I'll meet you later.
- DPO: Sure, I'll be back soon.
- Review: Both replace the food reminder with unrelated meeting/return updates.

### 2/6: ITEM-A-038 — LOCAL — tie

- Context: peer: are we still playing badminton tonight? → user: yeah, I think so
- Incoming: i might be ten mins late leh
- Supervised-only: okay, i'll be reaching around nine
- DPO: okay, i'll be there
- Review: Both give plausible user attendance/arrival updates; no fixed time distinguishes them.

### 2/7: ITEM-A-044 — STANDARD — both_bad

- Context: peer: have you submitted the project? → user: not yet, I am checking it first
- Incoming: remember ah, dont blur like sotong
- Supervised-only: sorry, I am blurred. I will check it later.
- DPO: Yes, I am blur. I will check it later.
- Review: Both use unnatural STANDARD blur/blurred phrasing rather than a clear reminder acknowledgment.

### 2/8: ITEM-A-046 — STANDARD — dpo

- Context: peer: want to study together tomorrow? → user: sure, what time?
- Incoming: 2pm u can anot, i can chiong from class
- Supervised-only: ok, I'll be there by 3pm.
- DPO: Ok, I'll be there at 2pm.
- Review: Directly accepts two; B changes to by three without framing a counterproposal.

### 2/9: ITEM-A-054 — LOCAL — tie

- Context: peer: We are going to the club later right? → user: Yeah, I will go back to shower first.
- Incoming: Aiyo, don't need la, let's go now
- Supervised-only: Okay, let's go now
- DPO: Ok, let's go now
- Review: Equivalent acceptance of going now.

### 2/10: ITEM-B-002 — STANDARD — tie

- Context: peer: I haven't eaten breakfast → user: same sia, I'm hungry also
- Incoming: where you want eat?
- Supervised-only: I'm going to eat at the cafeteria.
- DPO: I'm not sure yet.
- Review: Both a venue choice and uncertainty are valid selectable personal preferences.

### 2/11: ITEM-B-002 — LOCAL — tie

- Context: peer: I haven't eaten breakfast → user: same sia, I'm hungry also
- Incoming: where you want eat?
- Supervised-only: I'm going to eat at the hawker centre
- DPO: I'm not sure where to eat
- Review: Both choice and uncertainty are valid personal-preference drafts.

### 2/12: ITEM-B-009 — STANDARD — tie

- Context: peer: I'm going library now → user: I'll come over after class
- Incoming: still got space?
- Supervised-only: Yes, I'll be there in a few minutes.
- DPO: Yes, I'm going over now.
- Review: Both answer Yes and add a plausible personal arrival update.

### 2/13: ITEM-B-009 — LOCAL — dpo

- Context: peer: I'm going library now → user: I'll come over after class
- Incoming: still got space?
- Supervised-only: Okay, I'll come over after class
- DPO: Ya, still got space
- Review: Directly answers space availability; A repeats travel timing only.

### 2/14: ITEM-B-029 — LOCAL — tie

- Context: peer: I can't remember which room we're in → user: I'm checking the timetable
- Incoming: which room is our class in?
- Supervised-only: It's in the second floor
- DPO: Our class is in the second floor
- Review: Equivalent floor information in LOCAL.

### 2/15: ITEM-B-034 — STANDARD — both_bad

- Context: peer: I might not make it to the morning class → user: I can send you an update afterwards
- Incoming: can tell me what I miss?
- Supervised-only: Sure, let me know when you get your class scheduled.
- DPO: Sure, let me know what you missed.
- Review: A asks the absent peer what they missed instead of giving the promised update; B discusses class scheduling instead.

### 2/16: ITEM-B-043 — LOCAL — tie

- Context: peer: a few of us are going out this weekend → user: sounds fun
- Incoming: you want come?
- Supervised-only: okay, i'll come later
- DPO: yes, i'll come
- Review: Both accept the invitation with plausible timing.

### 2/17: ITEM-B-048 — LOCAL — tie

- Context: peer: are you coming to the event? → user: I was planning to
- Incoming: the event postpone for thirty minutes
- Supervised-only: Okay, I'll be there in a few minutes
- DPO: okay, I'll be there
- Review: Both acknowledge attending; a personal earlier arrival is allowed even if the event is delayed.

### 2/18: ITEM-B-050 — LOCAL — tie

- Context: peer: I haven't exercised in a while → user: same, I've been busy with classes
- Incoming: want to go run later?
- Supervised-only: Sure, I'll join you later.
- DPO: okay, I'll join later
- Review: Equivalent acceptance of joining the run.

### 2/19: ITEM-B-056 — STANDARD — tie

- Context: peer: are you free for badminton tonight? → user: I should be free
- Incoming: the court is already booked for seven
- Supervised-only: Okay, I'll be there by seven
- DPO: Okay, I'll be there.
- Review: Both confirm attending the established seven-o’clock plan.

### 2/20: ITEM-B-056 — LOCAL — tie

- Context: peer: are you free for badminton tonight? → user: I should be free
- Incoming: the court is already booked for seven
- Supervised-only: Okay, I'll be free for seven
- DPO: Okay, I'll be there
- Review: Both affirm availability/attendance for seven.

### 2/21: ITEM-C-001 — STANDARD — tie

- Context: peer: are you going to the library ltr? → user: yeah, probably after dinner
- Incoming: i'll be at lks around 8, want to meet there?
- Supervised-only: Sure, I'll meet you there around 8
- DPO: Sure, I'll meet you there.
- Review: Both accept the meeting; repeating eight is optional.

### 2/22: ITEM-C-001 — LOCAL — tie

- Context: peer: are you going to the library ltr? → user: yeah, probably after dinner
- Incoming: i'll be at lks around 8, want to meet there?
- Supervised-only: okay, i'll meet you there around 8
- DPO: okay, i'll meet you there
- Review: Both accept the meeting; repeating eight is optional.

### 2/23: ITEM-C-018 — STANDARD — dpo

- Context: peer: i'll upload my section before dinner → user: okay, then i can merge everything tonight
- Incoming: eh dont forget your slide ah, we need merge later
- Supervised-only: okay, i'll forget it later
- DPO: okay, i'll merge later.
- Review: Merging is the established action; forgetting contradicts the reminder.

### 2/24: ITEM-C-018 — LOCAL — dpo

- Context: peer: i'll upload my section before dinner → user: okay, then i can merge everything tonight
- Incoming: eh dont forget your slide ah, we need merge later
- Supervised-only: okay, i'll send you later
- DPO: okay, i'll merge later
- Review: Merging is clear and established; send you later lacks an object and clear action.

### 2/25: ITEM-C-023 — STANDARD — tie

- Context: peer: saturday im quite free → user: same, no plans yet
- Incoming: want me to jio you for the night market thing?
- Supervised-only: Sure, what time would you like to go?
- DPO: Sure, I'll be there.
- Review: Both acceptance and asking the useful next scheduling question are valid; no clear winner.

### 2/26: ITEM-C-023 — LOCAL — sft

- Context: peer: saturday im quite free → user: same, no plans yet
- Incoming: want me to jio you for the night market thing?
- Supervised-only: okay, i'll join you for the night market
- DPO: can, i jio you for the night market
- Review: Accepts joining the inviting peer; B unnecessarily reverses who is inviting whom.

### 2/27: ITEM-C-028 — LOCAL — sft

- Context: peer: im waiting at the cafe alr → user: coming now
- Incoming: paiseh, i ordered first because i was too hungry
- Supervised-only: okay, i'll be there soon
- DPO: okay, i'll wait for you at the cafe
- Review: User is coming and peer already at cafe; A preserves that arrangement more clearly than saying user will wait for peer.

### 2/28: ITEM-C-038 — STANDARD — dpo

- Context: peer: ive finished drafting the conclusion → user: great
- Incoming: can help me check if this part sounds too long?
- Supervised-only: Sure, it doesn't cost too much.
- DPO: Sure, I'll check it for you.
- Review: Offers the requested length check; A answers a nonexistent cost question.

### 2/29: ITEM-C-038 — LOCAL — tie

- Context: peer: ive finished drafting the conclusion → user: great
- Incoming: can help me check if this part sounds too long?
- Supervised-only: okay, let me check it for you
- DPO: okay, i'll check it later
- Review: Both offer the check, with different possible timing; no availability evidence selects one.

### 2/30: ITEM-C-041 — STANDARD — dpo

- Context: peer: could you upload your section before 8? → user: yes, i'll do it
- Incoming: paiseh, i only just finished class. i'll upload a bit later
- Supervised-only: ok, i'll send you later
- DPO: ok, i'll do it later
- Review: Do it can refer to the established own-section upload; send you later is malformed and unclear.

### 2/31: ITEM-C-041 — LOCAL — dpo

- Context: peer: could you upload your section before 8? → user: yes, i'll do it
- Incoming: paiseh, i only just finished class. i'll upload a bit later
- Supervised-only: okay, i'll send you later
- DPO: okay, i'll upload it later
- Review: Explicit upload action is clearer than send you later.

### 2/32: ITEM-C-045 — STANDARD — sft

- Context: peer: we said we'd sit together tmr → user: yep
- Incoming: you still coming early to get seats right?
- Supervised-only: Yes, I'm coming early to get seats.
- DPO: yes, I'm still waiting for seats.
- Review: Confirms early arrival for tomorrow; A says user is already waiting for seats.

### 2/33: ITEM-C-045 — LOCAL — sft

- Context: peer: we said we'd sit together tmr → user: yep
- Incoming: you still coming early to get seats right?
- Supervised-only: yep, i reaching early already
- DPO: yep, i still waiting for seats
- Review: Early arrival matches the requested plan better than an ambiguous current wait for seats.

### 2/34: ITEM-C-055 — STANDARD — tie

- Context: peer: lets go to the exhibition at 1 → user: sounds good
- Incoming: can go around 2 instead? i need lunch first
- Supervised-only: Sure, I'll be there around 2
- DPO: Sure, 2 works for me.
- Review: Both accept the proposed two-o’clock time.

### 2/35: ITEM-C-055 — LOCAL — tie

- Context: peer: lets go to the exhibition at 1 → user: sounds good
- Incoming: can go around 2 instead? i need lunch first
- Supervised-only: okay, 2 works for me
- DPO: can, 2 works for me
- Review: Equivalent LOCAL acceptance of two.

### 2/36: ITEM-C-058 — STANDARD — tie

- Context: peer: im heading to the court now → user: okay, i'll leave soon
- Incoming: court still occupied leh, need wait a bit
- Supervised-only: okay, i'll wait for you.
- DPO: okay, i'll wait for you
- Review: Punctuation-only difference in the same waiting plan.

### 2/37: ITEM-C-058 — LOCAL — sft

- Context: peer: im heading to the court now → user: okay, i'll leave soon
- Incoming: court still occupied leh, need wait a bit
- Supervised-only: okay, i'll leave soon
- DPO: okay, i'll wait for you
- Review: Keeps the user’s stated departure plan; B confusingly waits for the peer when the issue is court occupancy.

### 2/38: ITEM-D-007 — LOCAL — sft

- Context: peer: u wanna try the matcha at luckin? → user: nahh
- Incoming: then what u wanna drink later?
- Supervised-only: i wanna drink a bottle of beer
- DPO: i wanna drink later
- Review: Gives a plausible drink preference; A merely repeats wanting a drink later.

### 2/39: ITEM-D-010 — LOCAL — tie

- Context: peer: hello, can i ask u some question idk how to do → user: uhh i see if i can help
- Incoming: uhh, u know the guassian elim qn
- Supervised-only: i dont think i can help
- DPO: ya i know
- Review: Confirming knowledge and declining help for ability are both plausible personal-status choices.

### 2/40: ITEM-D-029 — STANDARD — both_bad

- Context: peer: eh so how, you still want bid for the morning class? → user: yah can
- Incoming: ok then rmb to bid before bidding ends
- Supervised-only: Okay, I'll see you later.
- DPO: okay, I'll send it to you after bidding ends.
- Review: Neither meaningfully acknowledges bidding before the deadline; both substitute an unrelated sending/farewell update.

### 2/41: ITEM-D-029 — LOCAL — both_bad

- Context: peer: eh so how, you still want bid for the morning class? → user: yah can
- Incoming: ok then rmb to bid before bidding ends
- Supervised-only: okay, i will send u the link
- DPO: okay, i will send you the link
- Review: Both introduce a link instead of responding to the bidding reminder.

### 2/42: ITEM-D-033 — STANDARD — sft

- Context: peer: hello → user: hi → peer: i am already at the bus stop
- Incoming: wya
- Supervised-only: where is the bus stop?
- DPO: okay, i will take you there
- Review: A location clarification can help coordinate meeting; A offers taking someone somewhere without answering or clarifying the current location.

### 2/43: ITEM-D-033 — LOCAL — sft

- Context: peer: hello → user: hi → peer: i am already at the bus stop
- Incoming: wya
- Supervised-only: okay, i reached the bus stop
- DPO: okay, i am reaching soon
- Review: Provides a plausible specific location in response to where; B gives only an arrival estimate.

### 2/44: ITEM-D-037 — STANDARD — sft

- Context: user: have you reached? → user: wya
- Incoming: i reached, i waiting outside mac
- Supervised-only: okay, i'll wait for you.
- DPO: okay, i'll take you there
- Review: Acknowledges arrival and gives a plausible waiting plan; A offers unrequested transport to an unspecified place.

### 2/45: ITEM-D-037 — LOCAL — tie

- Context: user: have you reached? → user: wya
- Incoming: i reached, i waiting outside mac
- Supervised-only: okay, i reached alr
- DPO: okay, i reached liao
- Review: Equivalent personal arrival status; particles do not establish a quality preference.

### 2/46: ITEM-D-038 — STANDARD — tie

- Context: peer: eh wya → user: i am outside mac → user: u eh
- Incoming: I am reaching soon, wait for me before going in.
- Supervised-only: ok, i will wait for you.
- DPO: ok, i will wait for you
- Review: Punctuation-only difference.

### 2/47: ITEM-D-038 — LOCAL — sft

- Context: peer: eh wya → user: i am outside mac → user: u eh
- Incoming: I am reaching soon, wait for me before going in.
- Supervised-only: ok, i will wait for you
- DPO: ok, i reaching soon
- Review: User already outside Mac should wait; A says user is arriving instead.

### 2/48: ITEM-D-050 — LOCAL — both_bad

- Context: user: eh u ok? why u never join us for bb session yesterday.
- Incoming: woah sorry, i came up with a flu.
- Supervised-only: okay, i'll join u later.
- DPO: okay, i'll join u later
- Review: Both give an unmotivated joining plan rather than respond to the peer’s explanation of illness.

### 2/49: ITEM-E-005 — STANDARD — sft

- Context: peer: i heard you going to cafe later to study → user: ya you want come? dont say i never jio
- Incoming: can ah, later i join you
- Supervised-only: ok, i will join you later
- DPO: ok, i join later
- Review: More grammatical STANDARD future phrasing; either user can plausibly arrive later, so this is not automatically a role contradiction.

### 2/50: ITEM-E-005 — LOCAL — tie

- Context: peer: i heard you going to cafe later to study → user: ya you want come? dont say i never jio
- Incoming: can ah, later i join you
- Supervised-only: can, i join you later
- DPO: can, i join later
- Review: Equivalent plausible joining updates in LOCAL.

### 2/51: ITEM-E-007 — LOCAL — both_bad

- Context: peer: tomorrow what you want to eat → user: lets eat at Chinatown tomorrow
- Incoming: onz, tmr meet na li
- Supervised-only: okay, tmr i can meet u all another time
- DPO: can, i meet u tmr
- Review: Neither answers the requested location; both drift into timing instead.

### 2/52: ITEM-E-009 — STANDARD — sft

- Context: peer: later I need to lend your charger to charge my phone → user: ok can but I can only give you after 3
- Incoming: hey can confirm i can use your charger after 3? thanks ah
- Supervised-only: Sure, I'll send you after 3
- DPO: Sure, I'll be able to use it after 3
- Review: Intended charger handoff matches the request; B shifts charger use to the user.

### 2/53: ITEM-E-009 — LOCAL — tie

- Context: peer: later I need to lend your charger to charge my phone → user: ok can but I can only give you after 3
- Incoming: hey can confirm i can use your charger after 3? thanks ah
- Supervised-only: okay, I'll be back after 3
- DPO: can, i confirm with you after 3
- Review: Both provide possible scheduling updates, though neither clearly confirms handoff now. Hold as equally weak rather than force a preference.

### 2/54: ITEM-E-025 — STANDARD — tie

- Context: peer: hi class, I had a family emergency and there will not be any class tomorrow. → user: oh no! what about the in class quiz?
- Incoming: will everyone be free on thursday instead for a make up class instead? paiseh
- Supervised-only: Sure, I'll be free on Thursday.
- DPO: Sure, I'll be free on thursday.
- Review: Capitalization-only difference.

### 2/55: ITEM-E-025 — LOCAL — tie

- Context: peer: hi class, I had a family emergency and there will not be any class tomorrow. → user: oh no! what about the in class quiz?
- Incoming: will everyone be free on thursday instead for a make up class instead? paiseh
- Supervised-only: okay, i'll be free on thursday
- DPO: yea, thursday is fine.
- Review: Equivalent Thursday availability confirmations.

### 2/56: ITEM-E-028 — LOCAL — both_bad

- Context: user: bro you want to eat at kopitiam later? The whole gang is cmg → user: if you want ah let us know
- Incoming: shag I damm busy sia, i cmi leh, dui bu qi
- Supervised-only: okay, I'll meet you at kopitiam later
- DPO: okay, i will meet you at kopitiam later
- Review: Both propose meeting a peer who explicitly cannot attend.

### 2/57: ITEM-E-032 — LOCAL — sft

- Context: peer: I reminded the class the tutorial is at SR3-1 liao → user: can you change today's tutorial to be in SR3-2 instead?
- Incoming: sorry ah but that one cannot, someone book liao
- Supervised-only: okay, let me know when the class starts
- DPO: okay, i booked already
- Review: Acknowledges the blocked venue and asks a relevant class-time follow-up; A says the unavailable booking is already done.

### 2/58: ITEM-E-040 — STANDARD — tie

- Context: peer: so much work this week wtf → user: ya sia
- Incoming: how ah, want meet together to do work?
- Supervised-only: yeah, I'll meet you there.
- DPO: Sure, I'll meet you there.
- Review: Equivalent acceptance of meeting to work.

### 2/59: ITEM-E-040 — LOCAL — tie

- Context: peer: so much work this week wtf → user: ya sia
- Incoming: how ah, want meet together to do work?
- Supervised-only: can, i meet y'all after work
- DPO: can, i meet yall after work
- Review: Apostrophe-only difference.

Identical outputs for this seed are preserved in the ignored blinded packet and raw run reports; equality is not an acceptability rating.

