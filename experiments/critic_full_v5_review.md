# Review of 94 retained training preferences — 2 November 2026

**All 94 retained preferences were inspected**, using each full conversation, the incoming message and both candidate replies. Raw pairwise judgments and isolated acceptance explanations were also inspected for flagged cases. This is an assistant review, not independent human gold. The user reviewed six policy examples, not these 94 labels.

## Outcome

- **73 keep:** a defensible preferred response and relative quality distinction.
- **3 exclude:** a clearly weak chosen target or a material reasoning/language-quality failure.
- **18 borderline:** plausible alternatives, ambiguous discourse/roles, weak ranking evidence or an unreviewed style boundary. Held out of the proposed training set.

The original 94-label report is unchanged. A separate proposed training file clears answers for excluded and borderline cases and retains every original raw critic decision. No excluded target is silently rewritten. No training was launched by this review.

## Concrete exclusions

| Comparison | Incoming message | Chosen draft | Review reason |
| --- | --- | --- | --- |
| 114 | we still playing bmt tonight? | Yes, we still have a lot to do. | The yes prefix is followed by an unrelated workload statement (we still have a lot to do). The critic invents a BMT simulation/game interpretation. Too weak and unnatural to use as the preferred STANDARD target. |
| 249 | walao eh 30 minutes so long, i zao first la | okay, I'll wait for you. | Peer says they will leave first (zao first), but the critic repeatedly interprets this as arriving first and rewards waiting for them. Exclude this departure/arrival misreading. |
| 338 | class start already? | Yes, it starts already. | The preferred STANDARD reply uses the unnatural tense it starts already. It answers the question better than the alternative, but is too weak as an absolute language-quality target. |

## Important interpretation

- Missing room/deadline/progress/approval evidence is **not** an exclusion reason. Those may be useful selectable drafts under the user’s policy.
- Keep does not mean perfect or the only valid answer. A preference between two acceptable replies can be useful when there is a clear wording/relevance distinction. Arbitrary personal-status or agree-versus-decline preferences are held out.
- A coherent new personal detail is not called a contradiction merely because it was not previously stated. Temporal progression may be plausible.
- The critic sometimes invents explanations or treats an acceptable refusal as inferior simply because it is less compliant. These are reasons to inspect labels rather than accept all 94 automatically.
- Three excludes are review decisions under the target quality standard, not a benchmark-derived critic error rate. Borderline cases are not declared objectively wrong.
- The source 94 labels cover 79 training conversations; the proposed subset is smaller still. They do not introduce new independent conversations.
- Fifteen conversations have retained preferences in both styles. Some differ in semantic content: ITEM-D-040 asks which movie in STANDARD but simply accepts in LOCAL; ITEM-B-013 acknowledges crowding in STANDARD but proposes another place in LOCAL. DPO on independent styles does not enforce paired semantic equivalence. Judge both-style outputs during validation.

## Full per-preference review

Each numbered comparison below includes the conversation and both replies, so the rationale is reviewable without opening the raw JSON.

### 1. ITEM-D-060 — STANDARD — KEEP

- Context: user: eh does the sch have printer
- Incoming: have but i think the printer down today sia
- Preferred: whats wrong with it ?
- Rejected: okay, let me check if there is a printer down
- Review: Relevant follow-up asks about the reported printer failure; alternative redundantly checks whether a printer is down.

### 8. ITEM-D-053 — STANDARD — KEEP

- Context: peer: eh, i got problem → user: wat problem
- Incoming: the printer down lol
- Preferred: no worries, what do you want to print?
- Rejected: okay, i will send you a message.
- Review: A useful printing follow-up is more responsive than an unspecified promise to send a message; both may be possible but the preference has a clear relevance basis.

### 10. ITEM-C-052 — LOCAL — KEEP

- Context: peer: im thinking of going to the club event tmr → user: i alr registered
- Incoming: can send me the sign-up link?
- Preferred: can, i send you the link
- Rejected: yes, the link goes to the link
- Review: Coherently agrees to share the sign-up link; rejected reply is a tautology.

### 11. ITEM-C-005 — LOCAL — KEEP

- Context: peer: yall going the workshop ah? → user: ya
- Incoming: yall tuesday no class ah?
- Preferred: ya, tuesday got class
- Rejected: okay i can
- Review: Answers the class-schedule question with a selectable personal detail; alternative does not answer it.

### 13. ITEM-C-022 — STANDARD — KEEP

- Context: peer: meet at exit b? → user: can
- Incoming: arrived, im standing near the escalator
- Preferred: okay, I'll meet you there.
- Rejected: im reaching a bit soon.
- Review: Natural acknowledgment and meeting commitment; alternative is awkwardly phrased.

### 18. ITEM-A-025 — STANDARD — KEEP

- Context: peer: Sorry, I will be about 15 minutes late → user: Hurry, I have been waiting for quite a while
- Incoming: paiseh, tahan abit, I reaching soon
- Preferred: Okay, I'll wait for you.
- Rejected: Okay, I'll come down soon
- Review: User is already waiting; waiting for the late peer preserves the roles better than coming down soon.

### 23. ITEM-B-001 — STANDARD — KEEP

- Context: peer: lecture ending soon → user: ya i hungry already sia
- Incoming: want go makan after this?
- Preferred: Sure, let's go after the lecture.
- Rejected: ya, I'll be back in my room.
- Review: Accepts the meal invitation at the proposed time; alternative changes to an unrelated room-status update.

### 28. ITEM-E-049 — LOCAL — KEEP

- Context: user: guys we need to finalize the slides → peer: i finished my part already → user: can we meet tonight to do a final check
- Incoming: ps i cmi tonight
- Preferred: can, i meet u tmr
- Rejected: im still on the train, we waiting for a meeting
- Review: Acknowledges tonight is unavailable and offers a later meeting; alternative is incoherent. A proposed new plan is allowed.

### 30. ITEM-C-036 — STANDARD — KEEP

- Context: peer: my notes for today are incomplete → user: i wrote down most of the examples
- Incoming: can send me the photo of that last example
- Preferred: sure, i'll send you the photo
- Rejected: can, I send you the picture as well
- Review: Directly offers to send the requested photo; alternative is less natural STANDARD and adds an unclear as well.

### 31. ITEM-D-028 — LOCAL — KEEP

- Context: peer: eh i am going to be late for class today.
- Incoming: can help me chope seat?
- Preferred: Can, i chope seat for you
- Rejected: I am leaving at 9
- Review: Agrees to reserve the requested seat; departure time alone does not address the request.

### 37. ITEM-C-010 — LOCAL — KEEP

- Context: peer: im heading to the library now → user: i came earlier to look for seats
- Incoming: got seat or not?
- Preferred: yea, i got a seat
- Rejected: ya, the student might have, the student one
- Review: Clear selectable seat-availability answer; alternative is incoherent.

### 38. ITEM-C-025 — STANDARD — KEEP

- Context: peer: im still stuck in the project room → user: im going to get a drink downstairs
- Incoming: can help me buy one iced coffee also?
- Preferred: Sure, I'll buy one for you.
- Rejected: sure, what would you like to buy?
- Review: Agrees to buy the already-specified iced coffee; alternative asks again what to buy.

### 42. ITEM-E-029 — LOCAL — KEEP

- Context: peer: i reaching in 15 mins, u there already? → user: yea i'm at the library already
- Incoming: bro help me chope some seats pls xie xie
- Preferred: okay, i chope a seat for you
- Rejected: oh my ah
- Review: Responds to the seat-reservation request; alternative does not meaningfully address it.

### 56. ITEM-A-027 — STANDARD — BORDERLINE

- Context: peer: we are meeting at Suntec city right? → user: Yep, on my way there now.
- Incoming: I a bit boliao, can i ask if ur taking MRT there?
- Preferred: Sure, I'll take MRT.
- Rejected: I'm sorry, MRT doesn't work today.
- Review: The pairwise critic rejects the MRT-status alternative using invented general-knowledge reasoning, although isolated judging accepts both. The preferred reply also changes a current-mode question into future tense. Weak rationale for a strong preference.

### 57. ITEM-C-010 — STANDARD — KEEP

- Context: peer: im heading to the library now → user: i came earlier to look for seats
- Incoming: got seat or not?
- Preferred: yes, i have a seat.
- Rejected: I think a seat available.
- Review: Grammatical, direct seat answer; alternative has a malformed clause. Unknown availability is allowed.

### 58. ITEM-D-031 — STANDARD — KEEP

- Context: peer: eh u also miss last week's class right? → user: ya
- Incoming: then should we still go for the makeup class
- Preferred: Sure, let's go for the makeup class.
- Rejected: Ok, let me see
- Review: Directly accepts the makeup-class proposal; alternative is a vague deferral.

### 65. ITEM-A-007 — STANDARD — KEEP

- Context: peer: The project work seems quite heavy. → user: Yeah, maybe we should start on it this weekend.
- Incoming: rabak, i only can meet next weekend
- Preferred: Okay, let's meet next weekend.
- Rejected: What are you planning to do?
- Review: Accommodates the stated next-weekend constraint; alternative asks an unrelated broad question.

### 66. ITEM-E-048 — LOCAL — KEEP

- Context: peer: idk what do, i'm kinda lost → user: don't worry, let me send you the plan for this week
- Incoming: wait u confirm this one legit ah?
- Preferred: yea, confirmed alr
- Rejected: yes, i confirm the plan confirmed
- Review: Natural confirmation instead of repeated confirm/confirmed phrasing; unknown plan legitimacy is allowed under selectable-draft policy.

### 69. ITEM-B-005 — STANDARD — KEEP

- Context: peer: want eat dinner at six? → user: six i cant sia
- Incoming: can we makan at seven instead?
- Preferred: Sure, seven works for me.
- Rejected: No, seven is fine.
- Review: Clearly agrees to seven; alternative combines contradictory No with seven is fine.

### 75. ITEM-B-022 — LOCAL — KEEP

- Context: peer: how is your section going? → user: almost finished
- Incoming: when do we need to submit this ah?
- Preferred: we need to submit this afternoon.
- Rejected: can! i go find a few weeks without taking my time
- Review: Provides a selectable deadline answer; rejected reply is incoherent. Missing deadline evidence is not treated as an error.

### 80. ITEM-C-015 — LOCAL — BORDERLINE

- Context: peer: i cant find yesterday's slides → user: i downloaded some files after class
- Incoming: you got the slides from yesterday?
- Preferred: yea, i got them from yesterday
- Rejected: nope i lost them
- Review: Downloading some files does not establish that these slides are available, and later loss is plausible. Both status alternatives can fit the selectable-draft policy; the critic falsely calls loss a contradiction.

### 83. ITEM-B-049 — STANDARD — KEEP

- Context: peer: I'll buy the tickets later → user: okay
- Incoming: can help me book one ticket also?
- Preferred: Sure, I'll book one for you.
- Rejected: Sure. I'll help you book one for you
- Review: Direct ticket-booking offer without the alternative’s repeated for you.

### 86. ITEM-C-021 — LOCAL — BORDERLINE

- Context: peer: im at the mrt alr → user: im walking from campus
- Incoming: you reach alr?
- Preferred: yea, i reach alr
- Rejected: ya, still at the mrt
- Review: Walking from campus and later reaching the MRT can both fit sequential turns. The critic treats location progression as an explicit contradiction. Preference is not sufficiently clear.

### 89. ITEM-E-045 — LOCAL — KEEP

- Context: user: hello i need your help to print something → user: u free?
- Incoming: sorry ah but i cannot leh
- Preferred: ok nvm eh
- Rejected: ok i send u the link
- Review: Acknowledges the peer cannot help; sending a link does not address the refusal.

### 90. ITEM-E-044 — LOCAL — KEEP

- Context: peer: what time we meeting again? → user: 4pm
- Incoming: 拜托 can meet 6pm instead?
- Preferred: Can, 6pm can
- Rejected: Okay. I ask my parents
- Review: Accepts the proposed new time; asking parents introduces a separate unclear action.

### 91. ITEM-D-055 — LOCAL — KEEP

- Context: peer: eh is my charger with you? → user: yah i am using it rn
- Incoming: okok, remember to return me
- Preferred: okay, i will return it later
- Rejected: Okay, I settle your own
- Review: Agrees to return the charger with correct actor and action; rejected reply is incoherent.

### 92. ITEM-B-039 — STANDARD — KEEP

- Context: peer: I'm near the wrong building → user: I know where you are
- Incoming: can come find me?
- Preferred: Sure, I'll find you where you are.
- Rejected: Okay, I'll wait for you while you are at the first level.
- Review: Offers the requested help finding the peer; alternative waits instead of going to find them.

### 95. ITEM-E-023 — LOCAL — KEEP

- Context: user: I'm abit lost for this week content i need help → peer: ok, will let prof know about your situation
- Incoming: eh prof want to organize a short tutorial class for you, you down?
- Preferred: sure eh
- Rejected: okay, i'll send someone to help me.
- Review: Accepts the offered tutorial; alternative has confused agency and fails to accept the offer clearly.

### 98. ITEM-D-025 — LOCAL — KEEP

- Context: peer: Hi hows your work going → user: good
- Incoming: Is it ok if i let you do the slides as well?
- Preferred: okay sure
- Rejected: okay, i will let you do it as well
- Review: User accepts doing slides; alternative reverses who will do them.

### 102. ITEM-D-044 — STANDARD — KEEP

- Context: peer: hello → user: yes?
- Incoming: rmb to pay me back for the food
- Preferred: Okay, I'll pay you back for it
- Rejected: Sure, would you be paying for the food?
- Review: Commits to repaying the peer; alternative asks whether the peer will pay instead.

### 111. ITEM-E-006 — LOCAL — BORDERLINE

- Context: peer: you going club today ah → user: ya later i going clubbing → user: you want come?
- Incoming: Paiseh but i cmi
- Preferred: okay nvm I'll go after work
- Rejected: okay, i can come later
- Review: Acknowledging the refusal and giving a personal plan is plausible, but calling the other personal-plan update a contradiction is too strong. Hold out this weak discourse preference.

### 114. ITEM-D-047 — STANDARD — EXCLUDE

- Context: peer: eh → user: yes?
- Incoming: we still playing bmt tonight?
- Preferred: Yes, we still have a lot to do.
- Rejected: Yes, gtg with me
- Review: The yes prefix is followed by an unrelated workload statement (we still have a lot to do). The critic invents a BMT simulation/game interpretation. Too weak and unnatural to use as the preferred STANDARD target.

### 116. ITEM-B-041 — STANDARD — KEEP

- Context: peer: I'm heading back to campus → user: I'm leaving soon also
- Incoming: want walk back together?
- Preferred: Sure, let's walk back together.
- Rejected: Yes, we can.
- Review: Natural, explicit acceptance of walking together; alternative is vaguer. Both acceptable, with a wording-based rather than personal-status preference.

### 118. ITEM-C-019 — STANDARD — KEEP

- Context: peer: rehearsal at 5 later? → user: yep, works for me
- Incoming: can push to 5:30? my class may end late
- Preferred: Sure, 5:30 is fine for me.
- Rejected: I think so
- Review: Explicitly accepts the requested revised time; alternative is uncertain and vague.

### 119. ITEM-A-015 — LOCAL — KEEP

- Context: peer: Hi would you be participating in our open session tonight? → user: Yes, where is it held at
- Incoming: Don't blur leh, it is at MPSH5
- Preferred: Okay, I'll meet you there
- Rejected: Oh, i feel like I was a bit out of the hall.
- Review: Acknowledges the stated venue and agrees to meet; alternative is incoherent.

### 122. ITEM-B-018 — STANDARD — BORDERLINE

- Context: peer: I couldn't attend yesterday's tutorial → user: I still got the recording
- Incoming: can share the recording with me?
- Preferred: Sure, I'll send you the link.
- Rejected: I'm sorry I can't help you.
- Review: The critic prefers agreeing to share over a permissible refusal. Without an established willingness to share, this risks teaching compliance rather than reply quality. Hold out rather than declare refusal wrong.

### 128. ITEM-E-023 — STANDARD — KEEP

- Context: user: I'm abit lost for this week content i need help → peer: ok, will let prof know about your situation
- Incoming: eh prof want to organize a short tutorial class for you, you down?
- Preferred: Yes, I'm down for it.
- Rejected: Yes, thanks, I have no idea
- Review: Directly accepts the tutorial offer; alternative tacks on an unclear no idea.

### 131. ITEM-A-040 — STANDARD — KEEP

- Context: peer: did you bring your charger? → user: yeah, why?
- Incoming: mine suddenly spoil sia
- Preferred: okay, I'll bring it for you.
- Rejected: Oh, it's so cold now. I am ready
- Review: Responds to the broken-charger issue with an offer using the charger user has; alternative is unrelated/incoherent.

### 137. ITEM-C-050 — LOCAL — KEEP

- Context: peer: im alr at the entrance → user: im on my way
- Incoming: paiseh, i walked to the wrong exit just now
- Preferred: okay, i come over now
- Rejected: okay, got it.
- Review: Offers to go over after the peer reports the wrong exit; generic acknowledgment is less helpful. The new prospective offer does not contradict context.

### 142. ITEM-E-011 — LOCAL — KEEP

- Context: peer: do you need help with printing your notes? i'm in the printer room now → user: yes, please wait ah i send you my notes to print. How much isit ah
- Incoming: abt $0.05 per page, quite cheap remember pay me can liao
- Preferred: okay, i pay you later
- Rejected: okay, np, i pay you when i pay you
- Review: Natural repayment commitment; alternative repeats pay without adding meaning.

### 152. ITEM-E-035 — STANDARD — KEEP

- Context: peer: I should be able to make it for the race next week! → user: sweet, see you there!
- Incoming: eh mb i cant make it liao i got something on
- Preferred: what is it?
- Rejected: okay, i'll be there in a few minutes
- Review: Relevant follow-up about the conflicting engagement; alternative says arriving in minutes for a next-week event and misses cancellation.

### 157. ITEM-D-055 — STANDARD — BORDERLINE

- Context: peer: eh is my charger with you? → user: yah i am using it rn
- Incoming: okok, remember to return me
- Preferred: Okay, i will return it for you
- Rejected: Okay, let me take it for you
- Review: Return it for you is understandable but less precise than return it to you; the role/preposition distinction is not handled reliably. Do not silently rewrite the chosen target.

### 160. ITEM-C-060 — LOCAL — KEEP

- Context: peer: we're supposed to meet at 6 for badminton → user: i'll be there
- Incoming: paiseh, my class ended late. maybe 15 minutes more
- Preferred: okay, i'll be there soon
- Rejected: okay, i meet you at 12 too
- Review: Acknowledges delay and gives a plausible personal arrival update; alternative introduces noon against the established six-o’clock plan.

### 162. ITEM-A-039 — STANDARD — KEEP

- Context: peer: Do you want to join us for dinner? → user: Sure! Where are y'all eating at?
- Incoming: we thinking to go food court makan
- Preferred: Okay, I'll meet you there.
- Rejected: No, I'll meet you there
- Review: Accepts the food-court plan; alternative combines No with an agreement.

### 167. ITEM-B-024 — STANDARD — KEEP

- Context: peer: I'm confused about the presentation → user: we can go through it together later
- Incoming: want discuss the project after class?
- Preferred: Sure, let's discuss it after class.
- Rejected: Yes, I will explain the final proposal.
- Review: Directly accepts the requested after-class discussion; alternative promises an explanation without answering the coordination as clearly.

### 168. ITEM-A-020 — STANDARD — KEEP

- Context: peer: Hey are you joining us for the party? → user: Yeah! Is it at the hall?
- Incoming: abuden, see you there!
- Preferred: Ok, see you there!
- Rejected: I see you there
- Review: Natural farewell/meeting acknowledgment; alternative has incorrect tense for the intended future meeting.

### 172. ITEM-A-025 — LOCAL — KEEP

- Context: peer: Sorry, I will be about 15 minutes late → user: Hurry, I have been waiting for quite a while
- Incoming: paiseh, tahan abit, I reaching soon
- Preferred: Okay, I'll wait for you.
- Rejected: I am reaching now, waiting for another 15 minutes
- Review: User already waiting, so continuing to wait preserves roles better than claiming user is arriving.

### 174. ITEM-D-040 — STANDARD — BORDERLINE

- Context: peer: eh are u free this sunday → user: yes
- Incoming: u want go watch movie
- Preferred: Sure, what movie is it?
- Rejected: Sure, why not?
- Review: Both Sure, what movie is it? and Sure, why not? are natural, useful responses. Asking an extra question is a subjective preference, not a clearly better answer.

### 177. ITEM-E-014 — STANDARD — BORDERLINE

- Context: peer: Lets meet at the station tonight → user: ok, chinatown mrt right?
- Incoming: yep, remember to bring the item i asked you
- Preferred: sorry, i didn't bring it yet.
- Rejected: I am on the way now.
- Review: Not bringing the item yet is a plausible status update but does not clearly respond to a reminder about a future meeting. Relative superiority over an on-the-way update is weak.

### 183. ITEM-D-040 — LOCAL — KEEP

- Context: peer: eh are u free this sunday → user: yes
- Incoming: u want go watch movie
- Preferred: can, i watch movie
- Rejected: no, can. i cfm can
- Review: Coherent acceptance of the movie invitation; alternative has confused polarity.

### 199. ITEM-D-049 — LOCAL — KEEP

- Context: peer: eh are u free tmr? → user: yes why?
- Incoming: want go run tmr
- Preferred: can, i join u tmr
- Rejected: ok, i can work thurs
- Review: Accepts the run invitation for tomorrow; alternative changes topic to working on Thursday.

### 200. ITEM-A-029 — STANDARD — KEEP

- Context: peer: Could you help me order chicken rice? → user: Okay, give me a second
- Incoming: peer: Wait, the chicken rice stall bo liao, noodles will do
- Preferred: Okay, I'll order for you.
- Rejected: I am okay with these
- Review: Continues the user’s food-ordering action after the peer changes to noodles; alternative vague personal approval does not specify helping.

### 202. ITEM-C-033 — STANDARD — KEEP

- Context: peer: i still need to organise all the lecture files → user: mine are a mess too
- Incoming: wanna compare our notes after class?
- Preferred: sure, we can compare them after class
- Rejected: Sure, I will do that for you
- Review: Explicitly accepts comparing notes after class; alternative is vague and can imply doing it on the peer’s behalf.

### 205. ITEM-C-003 — STANDARD — KEEP

- Context: peer: what do you wanna eat for lunch? → user: mala
- Incoming: tang or dry?
- Preferred: dry, please.
- Rejected: i prefer to get a steak
- Review: Answers dry-versus-soup choice for mala; alternative changes to steak without engaging the presented choice.

### 214. ITEM-B-060 — LOCAL — BORDERLINE

- Context: peer: can you lend me your adapter? → user: I was looking for it
- Incoming: sorry, I left the adapter at home
- Preferred: Which room u all at ah
- Rejected: Okay, I'll send you the adapter.
- Review: Asking which room the peer is in could coordinate help, but that intention is unstated and the source adapter exchange is itself confusing. Hold out without calling the draft inherently wrong.

### 215. ITEM-A-013 — STANDARD — BORDERLINE

- Context: peer: where are you? → user: I am going to buy a drink at the convenience store first.
- Incoming: eh bojio, I also want a drink
- Preferred: Okay, I'll get one at the convenience store.
- Rejected: Yes, I will get a glass of whisky.
- Review: The chosen drink reply can simply repeat the user’s own plan instead of clearly offering to include the peer. Both the comparison and intended beneficiary are ambiguous.

### 216. ITEM-D-006 — LOCAL — KEEP

- Context: peer: eh → user: what
- Incoming: Help me buy food eh
- Preferred: eh where u eating ah
- Rejected: I can't cook for you
- Review: Asks a plausible logistical follow-up to buying food; rejected reply refuses cooking, which was not requested.

### 219. ITEM-B-032 — STANDARD — KEEP

- Context: peer: can we meet before the tutorial? → user: sure, I'll be there at one thirty
- Incoming: can meet at two instead?
- Preferred: Sure, two works for me.
- Rejected: I'll be having two after that
- Review: Accepts the proposed two-o’clock revision; rejected reply is incoherent.

### 225. ITEM-B-008 — STANDARD — BORDERLINE

- Context: peer: I have quite a lot to finish today → user: same here
- Incoming: want to study at the library later?
- Preferred: Sure, let's meet at the library later.
- Rejected: sorry, the library is still full.
- Review: Accepting the library invitation and declining because it is full can both be valid selectable drafts. One rationale penalizes missing evidence despite the reviewed policy. Hold out this yes-versus-no preference.

### 232. ITEM-B-016 — LOCAL — KEEP

- Context: peer: I missed the first half of class → user: I was there from the start
- Incoming: they got upload the slides yet?
- Preferred: Yep, they uploaded the slides alr
- Rejected: Yeap, they got upload the slides yet
- Review: Provides a selectable upload-status answer; rejected reply echoes the question after Yep.

### 233. ITEM-B-018 — LOCAL — KEEP

- Context: peer: I couldn't attend yesterday's tutorial → user: I still got the recording
- Incoming: can share the recording with me?
- Preferred: Can, I send you the link
- Rejected: Can I share the message
- Review: Agrees to send the recording link; rejected reply asks an unclear question about sharing a message.

### 235. ITEM-C-033 — LOCAL — KEEP

- Context: peer: i still need to organise all the lecture files → user: mine are a mess too
- Incoming: wanna compare our notes after class?
- Preferred: yea, i think i can compare them after class
- Rejected: why?
- Review: Accepts comparing notes after class; bare why does not answer the invitation as helpfully.

### 239. ITEM-E-017 — LOCAL — KEEP

- Context: peer: lets meet at 6 later for a run! → user: aites see you later!
- Incoming: eh remember to meet later ah 6pm
- Preferred: okay, 6pm okay
- Rejected: eh 6pm why?
- Review: Confirms six, which was already agreed; alternative questions the agreed time without explanation.

### 241. ITEM-E-034 — LOCAL — KEEP

- Context: peer: hey lets go to this event together → user: Imma book the tickets
- Incoming: yep, remember to book the ticket hor
- Preferred: Okay, I booked the tickets alr
- Rejected: Ok eh, I book another one
- Review: Reports plausible booking progress; rejected reply suggests another ticket beyond the booking reminder.

### 249. ITEM-E-056 — STANDARD — EXCLUDE

- Context: peer: I taking MRT later, want meet? → user: um,  can wait for like 30 minutes
- Incoming: walao eh 30 minutes so long, i zao first la
- Preferred: okay, I'll wait for you.
- Rejected: no worries, wait for a moment
- Review: Peer says they will leave first (zao first), but the critic repeatedly interprets this as arriving first and rewards waiting for them. Exclude this departure/arrival misreading.

### 255. ITEM-E-033 — STANDARD — KEEP

- Context: peer: How are you man? its been a while → user: ya sia, what you doing this friday? u going cca?
- Incoming: bro u want come also? we got free bubble tea
- Preferred: Sure, I'll be there.
- Rejected: ok, can we ask u to come?
- Review: Accepts the CCA invitation; alternative reverses the invitation toward the peer.

### 256. ITEM-E-024 — STANDARD — KEEP

- Context: peer: woah is that a bot that generates notes for you? → user: yea, I made it myself and it has been a gamechanger sia
- Incoming: eh steng leh the auto generated notes your bot made for you, 我也要
- Preferred: okay, I'll send it to you later.
- Rejected: okay, I am free next time
- Review: Offers to share the requested bot/notes; alternative gives an unrelated availability update.

### 268. ITEM-A-036 — LOCAL — BORDERLINE

- Context: peer: What would you like from the drinks stall? → user: I would like a cold Milo
- Incoming: paiseh, help me to queue, i took the wrong queue number just now
- Preferred: Can, thank you
- Rejected: Okay, I'll send you the link
- Review: Can accepts the queueing request, but thank you is pragmatically awkward from the person being asked for help. Hold out the weak preferred target rather than promote it because the alternative mentions a link.

### 269. ITEM-E-052 — LOCAL — KEEP

- Context: peer: are we meeting this week? → user: yes, we are meeting at the MRT
- Incoming: eh can change to next week instead, 对不起
- Preferred: okay, we can change to next week
- Rejected: ya, thanks for reminding me. i was thinking of change
- Review: Clearly accepts moving the meeting to next week; alternative is unfinished and unclear.

### 275. ITEM-E-017 — STANDARD — KEEP

- Context: peer: lets meet at 6 later for a run! → user: aites see you later!
- Incoming: eh remember to meet later ah 6pm
- Preferred: okay, 6pm is fine.
- Rejected: Okay, 6pm still left
- Review: Confirms the established six-o’clock meeting; alternative is malformed.

### 280. ITEM-A-022 — STANDARD — BORDERLINE

- Context: peer: I am going to be late for the tutorial → user: Don't worry, I can reserve a seat for you
- Incoming: Can help me mark my attendance? Later kena marked down by prof
- Preferred: Sure, let me know when you reach.
- Rejected: No worries. Just go for a seat and find a seat
- Review: Sure may accept marking attendance, but let me know when you reach does not clearly specify the requested action. The critic overstates how directly it answers. Hold out the ambiguous target.

### 285. ITEM-C-051 — LOCAL — KEEP

- Context: peer: meet outside the library at 4 → user: sure
- Incoming: can meet at the mrt instead? its raining damn heavily here
- Preferred: okay, i meet you at the mrt
- Rejected: okay
- Review: Explicitly accepts moving to the MRT; bare okay is less informative. Preference is a clarity difference, not a claim that okay is invalid.

### 290. ITEM-E-018 — STANDARD — BORDERLINE

- Context: peer: omg is next week the quiz already → user: yea its quiz 1, lucky its open book so can bring cheatsheet → peer: wah I havent do up a cheatsheet yet
- Incoming: can we 一起 study and work on a cheatsheet tgt?
- Preferred: Sure, let's do it tgt.
- Rejected: okay, let me know when the lecture starts.
- Review: The preferred response is relevant, but tgt is shorthand in a STANDARD target. Whether STANDARD permits this abbreviation was not user-reviewed; hold out the style-boundary case.

### 292. ITEM-E-031 — STANDARD — BORDERLINE

- Context: peer: we can meet together next week saturday? → user: can ah I onz one
- Incoming: OMG i suddenly got something on that day, can meet another day?
- Preferred: Sure, another day?
- Rejected: Ok, afternoon also can.
- Review: Sure, another day? largely echoes the incoming request. It may be a valid confirmation question but is not a strong clearly phrased rescheduling target.

### 294. ITEM-C-057 — STANDARD — KEEP

- Context: peer: i forgot to bring anything to drink → user: im stopping by a shop first
- Incoming: can help me get one bottle of water?
- Preferred: Sure, I'll get one for you.
- Rejected: Yes, i will bring a free drink.
- Review: Agrees to get the requested water; alternative promises an unspecified free drink instead.

### 302. ITEM-C-051 — STANDARD — KEEP

- Context: peer: meet outside the library at 4 → user: sure
- Incoming: can meet at the mrt instead? its raining damn heavily here
- Preferred: okay, i'll meet you at the mrt
- Rejected: can, mrt i meet you at the library
- Review: Accepts the requested MRT location; alternative mixes MRT and library locations incoherently.

### 305. ITEM-C-052 — STANDARD — KEEP

- Context: peer: im thinking of going to the club event tmr → user: i alr registered
- Incoming: can send me the sign-up link?
- Preferred: Sure, I'll send you the link
- Rejected: sure, let me send you a link to send you
- Review: Natural link-sharing commitment; alternative repeats send/link without useful meaning.

### 314. ITEM-B-021 — STANDARD — BORDERLINE

- Context: peer: can share your notes from the lecture? → user: I have them open
- Incoming: can you send them now?
- Preferred: Sure, I'll send them later.
- Rejected: Sure, please send them by the library.
- Review: Sending later is a possible counterproposal to send now, but Sure obscures the time change. It is better than the incoherent alternative without being a strong preferred target.

### 320. ITEM-A-048 — LOCAL — KEEP

- Context: peer: sorry, I might be late for class today → user: how late will you be?
- Incoming: around 20 minutes sia, traffic very jialat
- Preferred: okay, let me know when you reach
- Rejected: ah.. i will be late because got driving class..
- Review: Responds to peer’s delay with an arrival check; alternative changes to user being late for a driving class instead.

### 324. ITEM-B-005 — LOCAL — KEEP

- Context: peer: want eat dinner at six? → user: six i cant sia
- Incoming: can we makan at seven instead?
- Preferred: can, seven works for me
- Rejected: ok i shall try to end 7 first
- Review: Clearly accepts seven; alternative unclear wording can imply finishing something at seven.

### 329. ITEM-C-004 — STANDARD — KEEP

- Context: peer: the work shop is till like 10 though → user: omg jesus
- Incoming: start from what time?
- Preferred: It starts at 8 o'clock.
- Rejected: You got the workshop one?
- Review: Provides a selectable start time; alternative does not answer when. No evidence requirement is imposed on the new time.

### 337. ITEM-B-024 — LOCAL — KEEP

- Context: peer: I'm confused about the presentation → user: we can go through it together later
- Incoming: want discuss the project after class?
- Preferred: ya can, we discuss after class.
- Rejected: can, we discuss after class after class
- Review: Accepts discussion after class without the alternative’s repeated after class.

### 338. ITEM-C-020 — STANDARD — EXCLUDE

- Context: peer: im walking over now → user: im near the classroom
- Incoming: class start already?
- Preferred: Yes, it starts already.
- Rejected: yea, i come and find you.
- Review: The preferred STANDARD reply uses the unnatural tense it starts already. It answers the question better than the alternative, but is too weak as an absolute language-quality target.

### 339. ITEM-A-029 — LOCAL — KEEP

- Context: peer: Could you help me order chicken rice? → user: Okay, give me a second
- Incoming: peer: Wait, the chicken rice stall bo liao, noodles will do
- Preferred: Okay, I'll order for you
- Rejected: Ya, I take the noodles stall for u to order, take us to cook noodles
- Review: Preserves the user’s role ordering food after substitution; alternative is incoherent and muddles cooking/ordering.

### 341. ITEM-A-040 — LOCAL — KEEP

- Context: peer: did you bring your charger? → user: yeah, why?
- Incoming: mine suddenly spoil sia
- Preferred: okay, i will bring mine for you
- Rejected: okok why leh
- Review: Offers the user’s charger to help with the peer’s broken one; alternative asks why despite an explanation already given.

### 347. ITEM-B-013 — LOCAL — KEEP

- Context: peer: where you studying today? → user: I haven't decided yet
- Incoming: the main library quite crowded today sia
- Preferred: okay, I'll go find another place.
- Rejected: I thought the library got crowded at all
- Review: Makes a plausible alternative study-place plan; rejected reply is ungrammatical and unclear.

### 350. ITEM-B-007 — STANDARD — BORDERLINE

- Context: peer: You at canteen already? → user: I walking there now
- Incoming: I'm there already
- Preferred: Okay, I'll wait for you.
- Rejected: Why don't you walk there and see your name?
- Review: The peer is already there and the user is walking over. I will wait for you may describe a later action, but it muddles who is waiting. Hold out rather than assert a definite temporal contradiction.

### 354. ITEM-B-053 — STANDARD — KEEP

- Context: peer: I haven't booked the court yet → user: let me know once you decide
- Incoming: can book the court for us?
- Preferred: Sure, I'll book it for you.
- Rejected: Sure, we can book a court for you
- Review: Natural court-booking commitment; alternative has less clear group/beneficiary phrasing.

### 357. ITEM-C-012 — LOCAL — KEEP

- Context: peer: study at 10 tmr? → user: ok, library?
- Incoming: 10 a bit early leh, 11 can?
- Preferred: can, 11 works for me
- Rejected: yah, 11 abit early
- Review: Clearly accepts eleven; alternative calls eleven early but does not answer as clearly.

### 363. ITEM-E-013 — STANDARD — BORDERLINE

- Context: peer: i got a room at library level 4 to work on our project → user: ok can thanks ah bro
- Incoming: yo rmb ah tmr library lv 4, u coming right?
- Preferred: yes, i am coming tmr
- Rejected: i'll be in abit later, will you see the link?
- Review: The confirmation fits the plan, but tmr is shorthand in a STANDARD target. The abbreviation policy is unreviewed; hold out rather than silently normalise the target.

### 365. ITEM-C-011 — LOCAL — KEEP

- Context: peer: im buying coffee downstairs → user: im alr at the library
- Incoming: can help me chope one seat first?
- Preferred: can, i chope one first
- Rejected: okay
- Review: Explicitly agrees to reserve the requested seat; bare acknowledgment is less clear. It is not treated as inherently invalid.

### 373. ITEM-B-045 — STANDARD — KEEP

- Context: peer: let's go to the museum on Saturday → user: Yea, Saturday can
- Incoming: are we still going to the museum this Saturday?
- Preferred: Yes, we are still going to the museum.
- Rejected: Nope, still haven't.
- Review: Confirms the previously agreed museum plan; alternative is an incomplete contradictory status answer.

### 376. ITEM-B-044 — STANDARD — KEEP

- Context: peer: there's a movie showing tonight → user: I haven't seen it yet
- Incoming: what time is the movie?
- Preferred: It's at seven o'clock.
- Rejected: it's at seven o'clock, which is at six
- Review: Provides a coherent selectable seven-o’clock time; rejected reply gives inconsistent six/seven times.

### 377. ITEM-B-013 — STANDARD — KEEP

- Context: peer: where you studying today? → user: I haven't decided yet
- Incoming: the main library quite crowded today sia
- Preferred: Yeah, it's very crowded.
- Rejected: Yes, I can find it.
- Review: Relevant acknowledgment of the library being crowded; rejected reply introduces an unclear ability to find it.

## Artifacts and next action

- Original: ignored `artifacts/critic_full_v5_screened.json`.
- Complete structured review: ignored `artifacts/critic_full_v5_assistant_review.json`.
- Proposed filtered sheet: ignored `artifacts/critic_full_v5_reviewed_proposed.json`.
- Review policy/source hash and individual decisions are preserved in both review artifacts.

Next prepare matched short supervised-only versus supervised+DPO runs using the proposed keep subset, with identical base/data/seeds and validation setup. The borderline cases need not block that conservative comparison. Do not promote this assistant review to human validation or claim model improvement before evaluation.

## Verification

The trainer loaded exactly **73** proposed preferences (37 STANDARD / 36 LOCAL), covering **61** training conversations. Verified all 380 original inputs against frozen training items, unchanged conversation/candidate text, original source hash, current training-file hash, and an isolated acceptable verdict for every retained winner. No both-bad reference was automatically promoted. No model weights were loaded or trained for this check, and no test data was opened.
