# Paired synthetic-data pilot — primary validation reply review

Seed **425**, fixed before inspecting new outputs. Arm names were hidden in the pair packet and labels recorded before opening this experiment’s key. However, the assistant authored the synthetic data and had seen control replies in the preceding DPO experiment, so this is **not fully blind or independent human evaluation**. Other seeds have quantitative checks, not additional qualitative ratings.

Across 100 style-specific outputs: control preferred **19**, augmentation **16**, tie **25**, both bad **10**, exact identical **30**. Identical outputs were not assumed good. The 10 both-bad ratings concern differing pairs only and are not an overall bad-output rate. Personal/status/factual details absent from context are allowed as selectable drafts.

The judgments below are subjective development review, not human gold. Each shows full context, both replies and the recorded rationale.

## 1. ITEM-A-010 — LOCAL — synthetic

- Context: peer: this course is so tough. → user: yeah, I could really use some help too.
- Incoming: peer: we very kiasu sia, first lesson already like that
- Control: okay, i will try to do it later.
- Augmented: yeah, i could use some help too.
- Review: Meaningfully echoes shared need for help; A introduces an unclear future action. Repetition alone does not make B wrong.

## 2. ITEM-A-017 — STANDARD — control

- Context: peer: Hey! Where are you now? → user: I am reaching Harbourfront MRT, how about you?
- Incoming: Eh i double check with you, we are meeting at 7pm right?
- Control: Yes, 7pm is fine.
- Augmented: Yes, I check with you at 7pm.
- Review: Confirms the meeting time; A changes meeting into checking with the peer at seven.

## 3. ITEM-A-024 — STANDARD — both_bad

- Context: peer: Hey have you printed the worksheet given out yesterday? → user: No, I will need to print it in school. Do you know how to use the printer?
- Incoming: paiseh, i don't know how to use the printer in school.
- Control: Okay, let me know how to use the printer.
- Augmented: Okay, I will print it in school. Do you know how to use the printer?
- Review: Both repeat an instruction request after the peer said they do not know how the printer works.

## 4. ITEM-A-024 — LOCAL — both_bad

- Context: peer: Hey have you printed the worksheet given out yesterday? → user: No, I will need to print it in school. Do you know how to use the printer?
- Incoming: paiseh, i don't know how to use the printer in school.
- Control: Okay, let me know how to use the printer.
- Augmented: Okay, I will print it in school. Do you know how to use the printer?
- Review: Both ask the peer for printer instructions despite the explicit lack of knowledge.

## 5. ITEM-A-030 — STANDARD — tie

- Context: peer: How long is our booking? → user: Oh no, our timer went off.
- Incoming: sian, need book somewhere else, our booking ran out
- Control: No worries, I'll book somewhere else.
- Augmented: Yes, we need to book somewhere else.
- Review: Both acknowledge/propose a new booking; offering to book and agreeing on the need are both valid drafts.

## 6. ITEM-A-030 — LOCAL — tie

- Context: peer: How long is our booking? → user: Oh no, our timer went off.
- Incoming: sian, need book somewhere else, our booking ran out
- Control: okay, we can book somewhere else
- Augmented: Okay, we can book somewhere else.
- Review: Capitalization and punctuation only.

## 7. ITEM-A-033 — STANDARD — tie

- Context: peer: Hey should we grab lunch? → user: sure! Lets head to Koufu
- Incoming: eh the chicken rice stall open today anot?
- Control: Yes, it opens today.
- Augmented: Yes, the chicken rice stall opens today.
- Review: Equivalent plausible opening-status answers.

## 8. ITEM-A-033 — LOCAL — control

- Context: peer: Hey should we grab lunch? → user: sure! Lets head to Koufu
- Incoming: eh the chicken rice stall open today anot?
- Control: yep, the chicken rice stall open today
- Augmented: eh, i think i can make it for dinner
- Review: Answers the stall-opening question; B changes topic to dinner availability.

## 9. ITEM-A-035 — STANDARD — both_bad

- Context: peer: I got to go for my appointment later → user: Yeah I should get going too
- Incoming: don't forget to dabao your food later
- Control: Okay, I'll be back soon.
- Augmented: Okay, I'll meet you later.
- Review: Neither addresses the takeaway-food reminder; both substitute a personal meeting/return update.

## 10. ITEM-A-038 — STANDARD — tie

- Context: peer: are we still playing badminton tonight? → user: yeah, I think so
- Incoming: i might be ten mins late leh
- Control: Okay, I'll be there around nine
- Augmented: Okay, I'll be there in ten minutes.
- Review: Both give possible user arrival times; no established user time makes one correct.

## 11. ITEM-A-044 — STANDARD — both_bad

- Context: peer: have you submitted the project? → user: not yet, I am checking it first
- Incoming: remember ah, dont blur like sotong
- Control: No worries, I will send it to you later.
- Augmented: No, I am blur. I am checking it first.
- Review: A promises sending to the peer instead of acknowledging submission; B uses unnatural STANDARD blur and repeats the old state.

## 12. ITEM-A-044 — LOCAL — synthetic

- Context: peer: have you submitted the project? → user: not yet, I am checking it first
- Incoming: remember ah, dont blur like sotong
- Control: okay, i will send it later
- Augmented: okay, I am checking it first.
- Review: Checking is the established project action; A gives an unclear sending update.

## 13. ITEM-A-046 — STANDARD — control

- Context: peer: want to study together tomorrow? → user: sure, what time?
- Incoming: 2pm u can anot, i can chiong from class
- Control: okay, 2pm sounds good.
- Augmented: okay, I'll meet you tomorrow.
- Review: Directly confirms the proposed two-o’clock time; B only confirms tomorrow.

## 14. ITEM-A-054 — LOCAL — control

- Context: peer: We are going to the club later right? → user: Yeah, I will go back to shower first.
- Incoming: Aiyo, don't need la, let's go now
- Control: Ok, let's go now
- Augmented: Okay, I'll be back soon.
- Review: Directly responds to going now; B changes to returning soon without explaining the plan.

## 15. ITEM-B-002 — STANDARD — control

- Context: peer: I haven't eaten breakfast → user: same sia, I'm hungry also
- Incoming: where you want eat?
- Control: I'm going to eat at the cafeteria.
- Augmented: I'm hungry now.
- Review: Answers where to eat with an allowed preference; B repeats hunger rather than a location.

## 16. ITEM-B-002 — LOCAL — control

- Context: peer: I haven't eaten breakfast → user: same sia, I'm hungry also
- Incoming: where you want eat?
- Control: I'm going to eat at the hawker centre
- Augmented: I'm hungry now
- Review: Answers where to eat; A only repeats hunger.

## 17. ITEM-B-009 — STANDARD — tie

- Context: peer: I'm going library now → user: I'll come over after class
- Incoming: still got space?
- Control: Yes, I'll be there in a few minutes.
- Augmented: No, I'll come over after class.
- Review: Both yes/no space answers can be selectable drafts and both include a plausible travel plan; unknown availability does not decide a preference.

## 18. ITEM-B-009 — LOCAL — control

- Context: peer: I'm going library now → user: I'll come over after class
- Incoming: still got space?
- Control: Ya, still got space
- Augmented: Okay, I'll come over after class
- Review: Directly answers space availability; A only repeats travel timing.

## 19. ITEM-B-029 — LOCAL — tie

- Context: peer: I can't remember which room we're in → user: I'm checking the timetable
- Incoming: which room is our class in?
- Control: It's in the second floor
- Augmented: I'm checking the timetable.
- Review: Continuing to check and providing floor information are both plausible drafts; do not penalize new location facts or force a status preference.

## 20. ITEM-B-034 — STANDARD — synthetic

- Context: peer: I might not make it to the morning class → user: I can send you an update afterwards
- Incoming: can tell me what I miss?
- Control: Sure, let me know what you missed.
- Augmented: Sure, I will send you an update after class.
- Review: Fulfils the promise to update the absent peer; A asks the absent peer to supply what was missed.

## 21. ITEM-B-034 — LOCAL — control

- Context: peer: I might not make it to the morning class → user: I can send you an update afterwards
- Incoming: can tell me what I miss?
- Control: Okay, let me know when you reach
- Augmented: Can, let me know what you miss
- Review: An arrival notification can coordinate giving the update; B reverses who is supposed to know what the absent peer missed. A remains only a partial reply.

## 22. ITEM-B-042 — STANDARD — both_bad

- Context: peer: I'm waiting outside already → user: I'm reaching soon
- Incoming: sorry, I took the wrong bus
- Control: I'm sorry, I didn't take it yet.
- Augmented: I'm sorry, I'll take the wrong bus.
- Review: B proposes taking the wrong bus; A gives an unclear transport apology rather than handling the peer’s problem.

## 23. ITEM-B-042 — LOCAL — synthetic

- Context: peer: I'm waiting outside already → user: I'm reaching soon
- Incoming: sorry, I took the wrong bus
- Control: Okay, I'll take the wrong bus.
- Augmented: I'm reaching soon
- Review: A plausible own-arrival update is better than proposing to take the wrong bus.

## 24. ITEM-B-043 — STANDARD — tie

- Context: peer: a few of us are going out this weekend → user: sounds fun
- Incoming: you want come?
- Control: Sure, I'll come.
- Augmented: Sure, I'm going to be a little late.
- Review: Both accept attendance with plausible personal timing.

## 25. ITEM-B-043 — LOCAL — tie

- Context: peer: a few of us are going out this weekend → user: sounds fun
- Incoming: you want come?
- Control: okay, i'll come later
- Augmented: I'm not sure if I can make it.
- Review: Accepting later and being uncertain about attendance are both allowed personal availability drafts.

## 26. ITEM-B-048 — STANDARD — control

- Context: peer: are you coming to the event? → user: I was planning to
- Incoming: the event postpone for thirty minutes
- Control: okay, I'll be there in about half an hour.
- Augmented: Yes, I'll be there for thirty minutes.
- Review: Addresses a half-hour postponement; A turns that delay into a stay duration without explaining it.

## 27. ITEM-B-048 — LOCAL — control

- Context: peer: are you coming to the event? → user: I was planning to
- Incoming: the event postpone for thirty minutes
- Control: okay, I'll be there
- Augmented: Okay, I'll be there for thirty minutes
- Review: Natural attendance acknowledgment; A appears to confuse the postponement interval with how long the user stays.

## 28. ITEM-B-050 — STANDARD — tie

- Context: peer: I haven't exercised in a while → user: same, I've been busy with classes
- Incoming: want to go run later?
- Control: Sure, I'll run later.
- Augmented: Sure, I'll join you later.
- Review: Both accept running together; minor wording difference.

## 29. ITEM-B-050 — LOCAL — tie

- Context: peer: I haven't exercised in a while → user: same, I've been busy with classes
- Incoming: want to go run later?
- Control: okay, I'll join later
- Augmented: okay, I'll join you later
- Review: Equivalent joining commitments.

## 30. ITEM-B-056 — STANDARD — control

- Context: peer: are you free for badminton tonight? → user: I should be free
- Incoming: the court is already booked for seven
- Control: Okay, seven works for me.
- Augmented: Okay, I'll be free for seven
- Review: More idiomatic STANDARD confirmation of seven; free for seven is awkward.

## 31. ITEM-B-056 — LOCAL — control

- Context: peer: are you free for badminton tonight? → user: I should be free
- Incoming: the court is already booked for seven
- Control: okay, seven works for me
- Augmented: Okay, I'll be free for seven
- Review: Clearer LOCAL agreement to seven than free for seven.

## 32. ITEM-B-057 — STANDARD — control

- Context: peer: my phone is almost out of battery → user: I brought my charger
- Incoming: can I borrow your charger?
- Control: Sure, I'll bring you a new one.
- Augmented: Sure, I can borrow your charger.
- Review: An offer to bring a charger can fit the request; B reverses who borrows whose charger.

## 33. ITEM-C-017 — STANDARD — synthetic

- Context: peer: im working on my slides now → user: same, im doing the results section
- Incoming: your part done alr?
- Control: yes, I am finished it alr.
- Augmented: Yes, I am done.
- Review: Grammatical STANDARD completion answer; B has broken grammar and local alr.

## 34. ITEM-C-017 — LOCAL — tie

- Context: peer: im working on my slides now → user: same, im doing the results section
- Incoming: your part done alr?
- Control: yea, i finished it alr
- Augmented: yea, i finished the slides alr
- Review: Both report completing the user’s slides/part; no clear quality difference.

## 35. ITEM-C-018 — LOCAL — tie

- Context: peer: i'll upload my section before dinner → user: okay, then i can merge everything tonight
- Incoming: eh dont forget your slide ah, we need merge later
- Control: okay, i'll merge later
- Augmented: okay, i can merge later
- Review: Both preserve the established merging task; a small modal difference is not decisive.

## 36. ITEM-C-023 — STANDARD — synthetic

- Context: peer: saturday im quite free → user: same, no plans yet
- Incoming: want me to jio you for the night market thing?
- Control: Sure, let me know when you're free.
- Augmented: Sure, I'll join you for the night market.
- Review: Directly accepts the night-market invitation; B re-asks availability already given.

## 37. ITEM-C-028 — STANDARD — synthetic

- Context: peer: im waiting at the cafe alr → user: coming now
- Incoming: paiseh, i ordered first because i was too hungry
- Control: okay, i'll wait for you at the cafe
- Augmented: okay, i'll take a look at the menu
- Review: Looking at the menu fits joining a peer who already ordered; B muddles who is waiting.

## 38. ITEM-C-038 — STANDARD — tie

- Context: peer: ive finished drafting the conclusion → user: great
- Incoming: can help me check if this part sounds too long?
- Control: Sure, let me check it for you.
- Augmented: Sure, it doesn't sound too long.
- Review: An immediate length assessment and an offer to check are both allowed selectable drafts; missing text evidence alone does not make the assessment wrong.

## 39. ITEM-C-041 — STANDARD — synthetic

- Context: peer: could you upload your section before 8? → user: yes, i'll do it
- Incoming: paiseh, i only just finished class. i'll upload a bit later
- Control: ok, i'll send you later
- Augmented: ok, i'll do it later.
- Review: A clearer reference to the upload task than the malformed send you later.

## 40. ITEM-C-041 — LOCAL — synthetic

- Context: peer: could you upload your section before 8? → user: yes, i'll do it
- Incoming: paiseh, i only just finished class. i'll upload a bit later
- Control: okay, i'll send you later
- Augmented: okay, i'll do it later
- Review: A clearer task commitment than the malformed send you later.

## 41. ITEM-C-045 — STANDARD — tie

- Context: peer: we said we'd sit together tmr → user: yep
- Incoming: you still coming early to get seats right?
- Control: Yes, I'm coming early to get seats.
- Augmented: yes, I'm coming early to get seats.
- Review: Capitalization-only difference.

## 42. ITEM-C-055 — STANDARD — tie

- Context: peer: lets go to the exhibition at 1 → user: sounds good
- Incoming: can go around 2 instead? i need lunch first
- Control: Sure, 2 works for me.
- Augmented: Sure, I'll meet you at 2
- Review: Equivalent acceptance of two.

## 43. ITEM-C-055 — LOCAL — tie

- Context: peer: lets go to the exhibition at 1 → user: sounds good
- Incoming: can go around 2 instead? i need lunch first
- Control: can, 2 works for me
- Augmented: can, i meet you at 2
- Review: Equivalent LOCAL acceptance of two.

## 44. ITEM-C-058 — STANDARD — synthetic

- Context: peer: im heading to the court now → user: okay, i'll leave soon
- Incoming: court still occupied leh, need wait a bit
- Control: okay, i'll wait for you.
- Augmented: okay, i'll leave soon
- Review: Preserves the user’s departure plan; B changes court occupancy into waiting for the peer.

## 45. ITEM-D-007 — LOCAL — control

- Context: peer: u wanna try the matcha at luckin? → user: nahh
- Incoming: then what u wanna drink later?
- Control: idk leh, i think i'll join later
- Augmented: i'm going to be a bit late
- Review: Idk at least addresses the drink choice; B only gives an unrelated lateness update.

## 46. ITEM-D-009 — STANDARD — both_bad

- Context: peer: eh i struggling to do the linear algebra qn → user: which
- Incoming: neh the gaussian elim qn
- Control: eh, i think i'll do it myself
- Augmented: eh, i'm going to do the algebra qn
- Review: Neither clearly engages the peer’s problem or offers help; both drift into the user doing their own algebra.

## 47. ITEM-D-009 — LOCAL — tie

- Context: peer: eh i struggling to do the linear algebra qn → user: which
- Incoming: neh the gaussian elim qn
- Control: eh i dont think i can do it
- Augmented: eh i dont think i can do the algebra qn
- Review: Equivalent possible ability-based refusal; specifying algebra does not change quality.

## 48. ITEM-D-010 — LOCAL — tie

- Context: peer: hello, can i ask u some question idk how to do → user: uhh i see if i can help
- Incoming: uhh, u know the guassian elim qn
- Control: ya i know
- Augmented: okay, i see if i can help
- Review: Confirming knowledge and offering to see whether one can help are both plausible; do not force a knowledge-status preference.

## 49. ITEM-D-029 — STANDARD — control

- Context: peer: eh so how, you still want bid for the morning class? → user: yah can
- Incoming: ok then rmb to bid before bidding ends
- Control: okay, thanks.
- Augmented: okay, I'll take it.
- Review: Acknowledges the bidding reminder; B introduces an unclear take action.

## 50. ITEM-D-029 — LOCAL — both_bad

- Context: peer: eh so how, you still want bid for the morning class? → user: yah can
- Incoming: ok then rmb to bid before bidding ends
- Control: okay, i will send u the link
- Augmented: okay, i'll send you a link
- Review: Both introduce a link instead of meaningfully responding to the bidding reminder.

## 51. ITEM-D-033 — STANDARD — synthetic

- Context: peer: hello → user: hi → peer: i am already at the bus stop
- Incoming: wya
- Control: okay, i will take a bus to get there
- Augmented: I am already at the bus stop
- Review: A plausible user location directly answers where; there is no earlier user location contradiction. B only gives a bus-travel plan.

## 52. ITEM-D-037 — STANDARD — control

- Context: user: have you reached? → user: wya
- Incoming: i reached, i waiting outside mac
- Control: okay, i'll wait for you.
- Augmented: okay, i'll take you there
- Review: Acknowledges arrival with a plausible waiting plan; A offers unrequested transport to an unspecified place.

## 53. ITEM-D-046 — LOCAL — both_bad

- Context: peer: eh help → user: what
- Incoming: the cinema ticket is sold out bruh
- Control: okay, i'll take it
- Augmented: okay, i'll take it for you
- Review: Neither clearly handles sold-out tickets; both use a generic take action with an unclear available object.

## 54. ITEM-D-050 — STANDARD — tie

- Context: user: eh u ok? why u never join us for bb session yesterday.
- Incoming: woah sorry, i came up with a flu.
- Control: oh, sorry, i didn't see u today.
- Augmented: oh, sorry, i couldn't help u.
- Review: Both are weak apologies rather than illness support; no clear comparative quality difference.

## 55. ITEM-D-050 — LOCAL — synthetic

- Context: user: eh u ok? why u never join us for bb session yesterday.
- Incoming: woah sorry, i came up with a flu.
- Control: sorry, i dont think i will be able to join us for bb session
- Augmented: okay, i'll see u later.
- Review: A plausible future visit/meeting is more coherent than apologizing as if the user missed the session.

## 56. ITEM-E-005 — STANDARD — synthetic

- Context: peer: i heard you going to cafe later to study → user: ya you want come? dont say i never jio
- Incoming: can ah, later i join you
- Control: ok, i join later
- Augmented: ok, i will join you later.
- Review: More grammatical STANDARD future phrasing; future user joining is not itself an established contradiction.

## 57. ITEM-E-005 — LOCAL — tie

- Context: peer: i heard you going to cafe later to study → user: ya you want come? dont say i never jio
- Incoming: can ah, later i join you
- Control: can, i join later
- Augmented: can, i join you later
- Review: Equivalent possible user joining updates.

## 58. ITEM-E-007 — STANDARD — both_bad

- Context: peer: tomorrow what you want to eat → user: lets eat at Chinatown tomorrow
- Incoming: onz, tmr meet na li
- Control: okay, tmr i meet u
- Augmented: okay, tmr i cannot
- Review: Both use shorthand in STANDARD and neither addresses the requested location.

## 59. ITEM-E-007 — LOCAL — tie

- Context: peer: tomorrow what you want to eat → user: lets eat at Chinatown tomorrow
- Incoming: onz, tmr meet na li
- Control: okay, tmr i meet u
- Augmented: tmr i cannot
- Review: A can cancel and B can confirm a plan; neither gives a location, but availability alone should not force a yes/no preference.

## 60. ITEM-E-009 — STANDARD — synthetic

- Context: peer: later I need to lend your charger to charge my phone → user: ok can but I can only give you after 3
- Incoming: hey can confirm i can use your charger after 3? thanks ah
- Control: Sure, I'll be able to use it after 3
- Augmented: okay, I'll send you after 3
- Review: Offers the intended charger handoff; A shifts charger use to the user.

## 61. ITEM-E-009 — LOCAL — synthetic

- Context: peer: later I need to lend your charger to charge my phone → user: ok can but I can only give you after 3
- Incoming: hey can confirm i can use your charger after 3? thanks ah
- Control: can, i confirm with you after 3
- Augmented: okay, I'll give you after 3
- Review: Directly offers the established after-three handoff; A defers confirmation without clearly answering the current request.

## 62. ITEM-E-015 — STANDARD — tie

- Context: peer: Hi guys, remember to do the quiz, deadline is next week → user: yep noted
- Incoming: hi have you done the quiz alr anot
- Control: No, I haven't done it yet.
- Augmented: Yes, I finished it last week.
- Review: Done and not-yet-done are both allowed personal-status drafts; a next-week deadline does not prove current progress.

## 63. ITEM-E-025 — STANDARD — tie

- Context: peer: hi class, I had a family emergency and there will not be any class tomorrow. → user: oh no! what about the in class quiz?
- Incoming: will everyone be free on thursday instead for a make up class instead? paiseh
- Control: yeah, thursday is fine.
- Augmented: sure, I'll be free on thursday.
- Review: Equivalent Thursday availability confirmations.

## 64. ITEM-E-027 — STANDARD — control

- Context: peer: eh the school got 3d printer anot how to use → user: have, i using it now, you need to book a slot to use → peer: ok i book later to use
- Incoming: paiseh ah can i change my booking i want to use it later
- Control: can, i change it later
- Augmented: sure, i can use it later
- Review: Changing the booking addresses the request; B changes who will use the printer.

## 65. ITEM-E-027 — LOCAL — control

- Context: peer: eh the school got 3d printer anot how to use → user: have, i using it now, you need to book a slot to use → peer: ok i book later to use
- Incoming: paiseh ah can i change my booking i want to use it later
- Control: can, i change it later
- Augmented: can, i use it later
- Review: Changes the booking as requested rather than the user’s own use time.

## 66. ITEM-E-028 — LOCAL — both_bad

- Context: user: bro you want to eat at kopitiam later? The whole gang is cmg → user: if you want ah let us know
- Incoming: shag I damm busy sia, i cmi leh, dui bu qi
- Control: okay, i can meet you at kopitiam later
- Augmented: okay, shall we meet at kopitiam later?
- Review: Both repeat the gathering plan without addressing the peer’s explicit inability to attend or stating a clear alternative time.

## 67. ITEM-E-032 — STANDARD — control

- Context: peer: I reminded the class the tutorial is at SR3-1 liao → user: can you change today's tutorial to be in SR3-2 instead?
- Incoming: sorry ah but that one cannot, someone book liao
- Control: okay, let me know when the class starts
- Augmented: okay, I'll book it for you.
- Review: Avoids promising the unavailable requested room; a class-time follow-up can be coherent.

## 68. ITEM-E-032 — LOCAL — synthetic

- Context: peer: I reminded the class the tutorial is at SR3-1 liao → user: can you change today's tutorial to be in SR3-2 instead?
- Incoming: sorry ah but that one cannot, someone book liao
- Control: okay, i booked for you
- Augmented: okay, let me know when the class starts
- Review: Avoids claiming the blocked booking was completed.

## 69. ITEM-E-040 — STANDARD — tie

- Context: peer: so much work this week wtf → user: ya sia
- Incoming: how ah, want meet together to do work?
- Control: yeah, I'll meet you there.
- Augmented: I'm not sure, I'll meet you sometime next week.
- Review: A counterproposal for next week and an acceptance are both valid personal scheduling drafts.

## 70. ITEM-E-040 — LOCAL — tie

- Context: peer: so much work this week wtf → user: ya sia
- Incoming: how ah, want meet together to do work?
- Control: can, i meet y'all after work
- Augmented: okay, i can meet you another day
- Review: Both give plausible alternative meeting times; no availability fact establishes a preference.

