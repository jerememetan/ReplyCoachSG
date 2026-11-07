# Revised reward scorer audit v2

## Summary

```json
{
  "regression_repaired_ids": [
    "ONPOLICY-006-S1"
  ],
  "regression_new_error_ids": [],
  "limitation": "Regression cases informed v2 prompt, not independent validation. Fresh48 replies are correlated within12 new same-family instances. Labels are assistant judgments, not human gold. Fresh gate unchanged, no post-result relabeling or prompt edits."
}
```

## fresh

```json
{
  "samples": 48,
  "scenario_groups": 12,
  "unique_prompt_drafts": 48,
  "expected_counts": {
    "acceptable": 11,
    "unacceptable": 24,
    "uncertain": 13
  },
  "positive_acceptance": 0.45454545454545453,
  "negative_rejection": 0.9583333333333334,
  "balanced_agreement": 0.7064393939393939,
  "false_positive_ids": [
    "ONPOLICY2-012-S2"
  ],
  "definite_abstentions": 0,
  "mixed_reward_groups": 6,
  "raw_failures": 0,
  "scoped_pilot_gate_pass": false,
  "scope": "Assistant judgments on48 correlated samples from12 fresh instances, not human accuracy or broad scorer readiness"
}
```

## regression_v1

```json
{
  "samples": 48,
  "scenario_groups": 12,
  "unique_prompt_drafts": 48,
  "expected_counts": {
    "acceptable": 14,
    "unacceptable": 23,
    "uncertain": 11
  },
  "positive_acceptance": 0.6428571428571429,
  "negative_rejection": 1.0,
  "balanced_agreement": 0.8214285714285714,
  "false_positive_ids": [],
  "definite_abstentions": 1,
  "mixed_reward_groups": 9,
  "raw_failures": 1,
  "scoped_pilot_gate_pass": false,
  "scope": "Assistant judgments on48 correlated samples from12 fresh instances, not human accuracy or broad scorer readiness"
}
```

## regression_v2

```json
{
  "samples": 48,
  "scenario_groups": 12,
  "unique_prompt_drafts": 48,
  "expected_counts": {
    "acceptable": 14,
    "unacceptable": 23,
    "uncertain": 11
  },
  "positive_acceptance": 0.7142857142857143,
  "negative_rejection": 1.0,
  "balanced_agreement": 0.8571428571428572,
  "false_positive_ids": [],
  "definite_abstentions": 0,
  "mixed_reward_groups": 9,
  "raw_failures": 0,
  "scoped_pilot_gate_pass": false,
  "scope": "Assistant judgments on48 correlated samples from12 fresh instances, not human accuracy or broad scorer readiness"
}
```

## All fresh replies

| Sample | Assistant verdict | Scorer verdict | Reply |
|---|---|---|---|
| ONPOLICY2-001-S1 | unacceptable | unacceptable | I cannot make it at five. Could we meet at two thirty? |
| ONPOLICY2-001-S2 | unacceptable | unacceptable | That works for me. See you then. |
| ONPOLICY2-001-S3 | unacceptable | unacceptable | That works for me. Let's meet at the front desk. |
| ONPOLICY2-001-S4 | uncertain | unacceptable | I cannot make it for the next day. Could we meet at the early morning? |
| ONPOLICY2-002-S1 | uncertain | unacceptable | I cannot stay at the cafe until tomorrow. Could we meet there after tomorrow? |
| ONPOLICY2-002-S2 | acceptable | acceptable | Sorry, I cannot stay there. Could we meet at the mrt instead? |
| ONPOLICY2-002-S3 | uncertain | unacceptable | I cannot stay there at a coffee shop instead. |
| ONPOLICY2-002-S4 | uncertain | unacceptable | I cannot stay there until tomorrow. Could I leave the cafe by 7:30? |
| ONPOLICY2-003-S1 | unacceptable | unacceptable | That works for me. Let's meet at the two-hour movie then. |
| ONPOLICY2-003-S2 | acceptable | unacceptable | No, I cannot do that. I can always start at two. |
| ONPOLICY2-003-S3 | unacceptable | unacceptable | I can't make it at the afternoon. |
| ONPOLICY2-003-S4 | acceptable | unacceptable | Yes, please, I can do it for tomorrow. |
| ONPOLICY2-004-S1 | unacceptable | unacceptable | No, I can't make it at five. |
| ONPOLICY2-004-S2 | unacceptable | unacceptable | No, I can't make it at the 9:30. |
| ONPOLICY2-004-S3 | uncertain | unacceptable | I can't make it at five. Could we meet another time? |
| ONPOLICY2-004-S4 | unacceptable | unacceptable | I can't make it at five thirty. Could we meet at eight? |
| ONPOLICY2-005-S1 | unacceptable | unacceptable | Friday is fine, let's meet then. |
| ONPOLICY2-005-S2 | unacceptable | unacceptable | Friday works for me. I can't do Thursday. |
| ONPOLICY2-005-S3 | uncertain | unacceptable | Friday works for me, let's meet next Friday. |
| ONPOLICY2-005-S4 | acceptable | unacceptable | Let's meet next Friday then. |
| ONPOLICY2-006-S1 | uncertain | acceptable | I can't make it. Could you bring a bottle of wine? |
| ONPOLICY2-006-S2 | unacceptable | unacceptable | Yes, I can come for the evening session. |
| ONPOLICY2-006-S3 | uncertain | unacceptable | Yes, I'll be there by the next session. |
| ONPOLICY2-006-S4 | unacceptable | unacceptable | I can't make it today, but I can leave at the booth. |
| ONPOLICY2-007-S1 | unacceptable | unacceptable | Yes, it's free. I have a shift on Sunday. |
| ONPOLICY2-007-S2 | acceptable | unacceptable | Sure, I will shift the shifts on Sunday. |
| ONPOLICY2-007-S3 | unacceptable | unacceptable | Yes, it's a shared one for my parents. |
| ONPOLICY2-007-S4 | uncertain | unacceptable | Yes, I have a shift on Sunday. |
| ONPOLICY2-008-S1 | unacceptable | unacceptable | Let's get it a coffee next Friday. |
| ONPOLICY2-008-S2 | uncertain | unacceptable | I cannot go with the following week, i will be back by then. |
| ONPOLICY2-008-S3 | uncertain | unacceptable | No, I am free on Friday. |
| ONPOLICY2-008-S4 | unacceptable | unacceptable | I cannot make it until Friday. Could we finish the order instead? |
| ONPOLICY2-009-S1 | unacceptable | unacceptable | Yes, I'll keep it to the cleaning area. |
| ONPOLICY2-009-S2 | unacceptable | unacceptable | Sorry, I would have gone to the other part of the jackets. |
| ONPOLICY2-009-S3 | unacceptable | unacceptable | I can't do tomorrow. Could we make it tomorrow? |
| ONPOLICY2-009-S4 | acceptable | acceptable | Yes, it's fine. |
| ONPOLICY2-010-S1 | acceptable | unacceptable | I'll return it to you at the class next time. |
| ONPOLICY2-010-S2 | acceptable | acceptable | I'll return it afternoon. |
| ONPOLICY2-010-S3 | unacceptable | unacceptable | No problem, I can return my bicycle tomorrow. |
| ONPOLICY2-010-S4 | acceptable | unacceptable | I can return your bicycle during class. |
| ONPOLICY2-011-S1 | uncertain | unacceptable | Yes, I still have them, but I can't find them. |
| ONPOLICY2-011-S2 | acceptable | acceptable | Yes, they are still in my backpack. |
| ONPOLICY2-011-S3 | unacceptable | unacceptable | Yes, I have mine. |
| ONPOLICY2-011-S4 | acceptable | acceptable | Yes, I have them in my backpack. |
| ONPOLICY2-012-S1 | uncertain | unacceptable | I'm already waiting for it to bring back. |
| ONPOLICY2-012-S2 | unacceptable | acceptable | Yes, I could bring your cable back tonight. |
| ONPOLICY2-012-S3 | unacceptable | unacceptable | I'd be happy to bring it. |
| ONPOLICY2-012-S4 | unacceptable | unacceptable | Yes, I will bring my cable tonight. |

### Disagreements

**ONPOLICY2-006-S3** · Context ['Is the evening session okay?', 'No, I can attend only the lunchtime session.'] · Incoming: Come to the evening session with me?
Draft: Yes, I'll be there by the next session.
Assistant: uncertain — Next session might be lunchtime or evening; unspecified session prevents confident scoring.
Scorer: unacceptable — The user explicitly stated they can only attend the lunchtime session, but the draft agrees to attend the evening session.

**ONPOLICY2-003-S2** · Context ['I need to be home no later than22:15.'] · Incoming: Want to start a two-hour film at21:30?
Draft: No, I cannot do that. I can always start at two.
Assistant: acceptable — Refuses requested time and proposes two, an earlier time;2pm or2am can avoid21:30 film conflict.
Scorer: unacceptable — The draft proposes starting a two-hour film at 2:00, which would end at 4:00, but it does not address the user's constraint of needing to be home by 22:15. More importantly, the draft is logically inconsistent with the user's stated constraint as a viable solution for the current request.

**ONPOLICY2-002-S3** · Context ['I have to leave the cafe by14:10 to catch my bus.'] · Incoming: Can you stay at the cafe until14:45?
Draft: I cannot stay there at a coffee shop instead.
Assistant: uncertain — Looks like a refusal but at a coffee shop instead is malformed and obscures intended alternative.
Scorer: unacceptable — The draft provides a non-sequitur response that does not address the request to stay until a specific time, instead making an irrelevant statement about the location type.

**ONPOLICY2-006-S1** · Context ['Is the evening session okay?', 'No, I can attend only the lunchtime session.'] · Incoming: Come to the evening session with me?
Draft: I can't make it. Could you bring a bottle of wine?
Assistant: uncertain — Refusal is valid but wine request is an unexplained tangent; uncertain overall conversational adequacy.
Scorer: acceptable — The user explicitly refused the evening session in the previous turn and maintains that refusal in the draft. The request for wine is a new, compatible request.

**ONPOLICY2-007-S2** · Context ['I have a shift for the whole of Sunday. Saturday is free.'] · Incoming: Fancy a day trip on Sunday?
Draft: Sure, I will shift the shifts on Sunday.
Assistant: acceptable — Explicitly proposes shifting work shifts to attend. Shift was not stated immovable; a compatible changed personal plan can be a selectable draft.
Scorer: unacceptable — The user explicitly stated they have a shift for the whole of Sunday, but the draft suggests they will 'shift the shifts' to accommodate the trip, which contradicts their initial statement of having a shift.

**ONPOLICY2-002-S4** · Context ['I have to leave the cafe by14:10 to catch my bus.'] · Incoming: Can you stay at the cafe until14:45?
Draft: I cannot stay there until tomorrow. Could I leave the cafe by 7:30?
Assistant: uncertain — Departure7:30 could respect14:10 maximum, but tomorrow reference and unspecified morning/evening make intent unclear.
Scorer: unacceptable — The draft contains an internal contradiction by suggesting a departure time of 7:30 (which is earlier than the previously stated 14:10) while simultaneously claiming they cannot stay until 'tomorrow', which is a non-sequitur to the current request about staying until 14:45.

**ONPOLICY2-001-S4** · Context ['What time does your class finish?', 'It finishes at17:20. I cannot meet before17:20.'] · Incoming: Could we meet at16:45?
Draft: I cannot make it for the next day. Could we meet at the early morning?
Assistant: uncertain — Cannot next day then proposes early morning; which day and relation to current request are unclear.
Scorer: unacceptable — The user's draft proposes a meeting in the 'early morning' without specifying a time, and fails to address the specific request for 16:45 while ignoring the previously stated constraint of not being able to meet before 17:20.

**ONPOLICY2-005-S3** · Context ['I cannot meet this Friday. I am free next Monday instead.'] · Incoming: Can we do this Friday?
Draft: Friday works for me, let's meet next Friday.
Assistant: uncertain — Initial Friday acceptance then next Friday could mean distinct following week; ambiguous this-vs-next commitment.
Scorer: unacceptable — The draft contradicts the user's explicit statement that they cannot meet this Friday.

**ONPOLICY2-002-S1** · Context ['I have to leave the cafe by14:10 to catch my bus.'] · Incoming: Can you stay at the cafe until14:45?
Draft: I cannot stay at the cafe until tomorrow. Could we meet there after tomorrow?
Assistant: uncertain — Tomorrow/after tomorrow might be an alternative date, but cannot stay until tomorrow misstates the14:45 request.
Scorer: unacceptable — The draft introduces a contradiction by suggesting a meeting 'after tomorrow' when the original context was a same-day cafe visit, and it fails to address the specific request of staying until 14:45 by proposing a completely different timeframe without a logical bridge.

**ONPOLICY2-008-S2** · Context ['I am away all next week. I will be back the following Monday.'] · Incoming: Can we get coffee next Thursday?
Draft: I cannot go with the following week, i will be back by then.
Assistant: uncertain — Following-week wording may conflict with or differ from return Monday; intent unclear rather than definite refusal of Thursday.
Scorer: unacceptable — The draft is incoherent and does not provide a clear answer to the request for coffee next Thursday. It fails to address the specific day requested and contains a grammatical error ('I cannot go with the following week').

**ONPOLICY2-004-S3** · Context ['The earliest I can arrive is08:40.'] · Incoming: Are you able to get here at08:15?
Draft: I can't make it at five. Could we meet another time?
Assistant: uncertain — Another-time request could be valid but the five reference misstates08:15; partial relevance without clear commitment.
Scorer: unacceptable — The draft provides a non-answer to the specific time requested (08:15) by referencing a completely different time (five) that was never mentioned in the conversation.

**ONPOLICY2-007-S4** · Context ['I have a shift for the whole of Sunday. Saturday is free.'] · Incoming: Fancy a day trip on Sunday?
Draft: Yes, I have a shift on Sunday.
Assistant: uncertain — Affirms having Sunday shift but yes could accept hiking; implied refusal versus acceptance is ambiguous.
Scorer: unacceptable — The user explicitly stated they have a shift on Sunday, making a day trip impossible. The draft incorrectly accepts the invitation despite the stated unavailability.

**ONPOLICY2-012-S1** · Context ['You borrowed my charging cable for your trip.'] · Incoming: Should I bring your cable back tonight?
Draft: I'm already waiting for it to bring back.
Assistant: uncertain — Waiting for return could be correct owner role, but malformed bring back wording makes intended actor unclear.
Scorer: unacceptable — The draft contains a logical contradiction regarding ownership and possession. The user previously stated that the peer borrowed the cable (peer holds it), but the draft implies the user is waiting for the peer to bring it back while simultaneously suggesting the user is 'waiting for it' in a way that confuses the roles of owner and holder.

**ONPOLICY2-010-S4** · Context ['I borrowed your bicycle to get to class.'] · Incoming: Please return my bicycle this afternoon.
Draft: I can return your bicycle during class.
Assistant: acceptable — Borrower correctly names your bicycle and offers return during class, a valid timing proposal.
Scorer: unacceptable — The user's draft proposes returning the bicycle during class, which contradicts the initial statement that they borrowed the bicycle specifically to get to class (implying they are currently in class or occupied by it). More importantly, it fails to address the specific timeframe requested ('this afternoon') by offering a different, potentially conflicting situation.

**ONPOLICY2-011-S1** · Context ['Have you seen my keys?', 'Yes, your keys are in my backpack.'] · Incoming: Do you still have my keys with you?
Draft: Yes, I still have them, but I can't find them.
Assistant: uncertain — Having keys but not finding them can coexist but is confusing relative to known backpack location; no definite ownership reversal.
Scorer: unacceptable — The draft contains an internal contradiction: it claims to still have the keys while simultaneously stating it cannot find them.

**ONPOLICY2-008-S3** · Context ['I am away all next week. I will be back the following Monday.'] · Incoming: Can we get coffee next Thursday?
Draft: No, I am free on Friday.
Assistant: uncertain — Friday could mean this Friday or next Friday as alternative to next Thursday; no date given.
Scorer: unacceptable — The user explicitly stated they are away all next week, but the draft claims they are free on Friday (which falls within that week).

**ONPOLICY2-005-S4** · Context ['I cannot meet this Friday. I am free next Monday instead.'] · Incoming: Can we do this Friday?
Draft: Let's meet next Friday then.
Assistant: acceptable — Explicit next-Friday counterproposal is distinct from this-Friday request; Monday availability does not prohibit other future days.
Scorer: unacceptable — The user explicitly stated they are free next Monday, but the draft proposes next Friday, which contradicts the user's stated availability.

**ONPOLICY2-010-S1** · Context ['I borrowed your bicycle to get to class.'] · Incoming: Please return my bicycle this afternoon.
Draft: I'll return it to you at the class next time.
Assistant: acceptable — Borrower promises to return peer bicycle at class next time; compatible counterproposal to afternoon request.
Scorer: unacceptable — The user's draft proposes a different time/location (next class) without acknowledging or addressing the specific request to return it 'this afternoon'.

**ONPOLICY2-003-S4** · Context ['I need to be home no later than22:15.'] · Incoming: Want to start a two-hour film at21:30?
Draft: Yes, please, I can do it for tomorrow.
Assistant: acceptable — Explicit tomorrow counterproposal shifts event off tonight; future availability is a permitted selectable detail.
Scorer: unacceptable — The user's draft accepts a film starting at 21:30 for two hours, which would end at 23:30, violating their stated constraint of needing to be home by 22:15.

**ONPOLICY2-012-S2** · Context ['You borrowed my charging cable for your trip.'] · Incoming: Should I bring your cable back tonight?
Draft: Yes, I could bring your cable back tonight.
Assistant: unacceptable — Owner takes peer returner role and calls object your cable.
Scorer: acceptable — The user correctly identifies their role as the borrower and agrees to return the cable tonight as requested.

## All regression replies

| Sample | Assistant verdict | Scorer verdict | Reply |
|---|---|---|---|
| ONPOLICY-001-S1 | unacceptable | unacceptable | Sorry, 5:30 works for me. |
| ONPOLICY-001-S2 | unacceptable | unacceptable | Yes, 6:30 works for me. |
| ONPOLICY-001-S3 | acceptable | acceptable | I cannot make it tonight, sorry. |
| ONPOLICY-001-S4 | uncertain | unacceptable | Let's meet at the mrt at the place. |
| ONPOLICY-002-S1 | unacceptable | unacceptable | I would love to, but I can't finish it by seven. |
| ONPOLICY-002-S2 | acceptable | unacceptable | I cannot make it tonight. Could we meet at the two-hour movie tomorrow then? |
| ONPOLICY-002-S3 | uncertain | unacceptable | Sorry, I cannot make it at the late. |
| ONPOLICY-002-S4 | uncertain | unacceptable | That works for me. Let's meet at the hotel at 5:00. |
| ONPOLICY-003-S1 | uncertain | acceptable | I cannot join, can you reach in a few minutes? |
| ONPOLICY-003-S2 | unacceptable | unacceptable | I cannot be there by tomorrow. |
| ONPOLICY-003-S3 | unacceptable | unacceptable | I cannot make it for a meeting at the evening, but I can go with another speaker. |
| ONPOLICY-003-S4 | uncertain | unacceptable | Sorry, I cannot come for the evening. Could we meet after that? |
| ONPOLICY-004-S1 | acceptable | unacceptable | Yes, tomorrow evening works for me. |
| ONPOLICY-004-S2 | unacceptable | unacceptable | I can't leave after eight, but I can only leave around three. |
| ONPOLICY-004-S3 | uncertain | acceptable | I cannot stay until eleven. |
| ONPOLICY-004-S4 | uncertain | unacceptable | I cannot stay until six. Could we meet at the gate on Sunday? |
| ONPOLICY-005-S1 | uncertain | unacceptable | Let's meet at the same time as Wednesday evening. |
| ONPOLICY-005-S2 | unacceptable | unacceptable | Wednesday evening is fine. |
| ONPOLICY-005-S3 | unacceptable | unacceptable | Wednesday evening works for me. Wednesday evening works for me. |
| ONPOLICY-005-S4 | unacceptable | unacceptable | Wednesday evening works for me. I am free on Thursday evening. |
| ONPOLICY-006-S1 | acceptable | acceptable | I cannot, I cannot go anywhere else. |
| ONPOLICY-006-S2 | acceptable | unacceptable | I cannot attend it during the afternoon, but I cannot attend the afternoon session. |
| ONPOLICY-006-S3 | unacceptable | unacceptable | No worries, I will join the afternoon session. |
| ONPOLICY-006-S4 | acceptable | acceptable | I cannot join the afternoon session. I am busy with my work. |
| ONPOLICY-007-S1 | unacceptable | unacceptable | Yes, Saturday sounds good. Let's go to the place then. |
| ONPOLICY-007-S2 | unacceptable | unacceptable | Sure, I will meet you at the hiking spots. |
| ONPOLICY-007-S3 | acceptable | acceptable | Sorry, I can't make it. How about the next time? |
| ONPOLICY-007-S4 | unacceptable | unacceptable | Yes, Saturday is fine. |
| ONPOLICY-008-S1 | acceptable | acceptable | I can't make it, sorry. |
| ONPOLICY-008-S2 | acceptable | acceptable | Sorry, I am not free this weekend. Could we meet next week? |
| ONPOLICY-008-S3 | unacceptable | unacceptable | Yes, let's meet for a lunch. |
| ONPOLICY-008-S4 | uncertain | acceptable | I cannot make it for lunch today. |
| ONPOLICY-009-S1 | acceptable | acceptable | Yes, you can return it next week. |
| ONPOLICY-009-S2 | acceptable | acceptable | Yes, please return it next week. |
| ONPOLICY-009-S3 | unacceptable | unacceptable | I can't return it next week. |
| ONPOLICY-009-S4 | unacceptable | unacceptable | Yes, I can return my racket this afternoon. |
| ONPOLICY-010-S1 | unacceptable | unacceptable | I cannot bring mine back today, but I can keep it as tomorrow morning. |
| ONPOLICY-010-S2 | acceptable | acceptable | I can't bring it back for you tomorrow. |
| ONPOLICY-010-S3 | uncertain | acceptable | I can't do it tomorrow. Could you use my laptop? |
| ONPOLICY-010-S4 | unacceptable | unacceptable | Sure, I'll bring my power bank tomorrow. |
| ONPOLICY-011-S1 | unacceptable | unacceptable | Yes, I have my library card in my wallet. |
| ONPOLICY-011-S2 | uncertain | unacceptable | I have it now, but I have it last night. |
| ONPOLICY-011-S3 | acceptable | unacceptable | I have it, but I have to show you your library card in my wallet. |
| ONPOLICY-011-S4 | acceptable | acceptable | Yes, I do. It has already been added to my wallet. |
| ONPOLICY-012-S1 | unacceptable | unacceptable | I'd like to return my chair today. |
| ONPOLICY-012-S2 | unacceptable | unacceptable | I can't return my chair since it's empty. |
| ONPOLICY-012-S3 | unacceptable | unacceptable | I'm sorry, but I'll head to another dining room. |
| ONPOLICY-012-S4 | unacceptable | unacceptable | I'd be happy to return it, but I can't return it. |

### Disagreements

**ONPOLICY-005-S1** · Context ['I am unavailable on Wednesday. Thursday evening is free.'] · Incoming: Can we meet on Wednesday evening?
Draft: Let's meet at the same time as Wednesday evening.
Assistant: uncertain — Same time as Wednesday evening could mean Wednesday itself or another day at the same clock time; cannot award a definite positive.
Scorer: unacceptable — The user explicitly stated they are unavailable on Wednesday, but the draft proposes meeting on Wednesday evening.

**ONPOLICY-011-S2** · Context ['I have your library card in my wallet.'] · Incoming: Do you still have my library card?
Draft: I have it now, but I have it last night.
Assistant: uncertain — Affirms possession but last-night clause is malformed and contrast unclear; do not conflate tense error with definite role reversal.
Scorer: unacceptable — The draft contains an internal contradiction/logical error ('I have it now, but I have it last night') and fails to provide a clear answer regarding the current status of the card.

**ONPOLICY-008-S4** · Context ['I am not free this weekend. I am free next Tuesday.'] · Incoming: Shall we have lunch this Sunday?
Draft: I cannot make it for lunch today.
Assistant: uncertain — Declines lunch today instead of Sunday; context does not establish today is Sunday.
Scorer: acceptable — The user is refusing the Sunday lunch request, which is consistent with their stated unavailability for the weekend.

**ONPOLICY-002-S3** · Context ['I need to be home by 21:00 tonight.'] · Incoming: Shall we start our two-hour movie at 20:00?
Draft: Sorry, I cannot make it at the late.
Assistant: uncertain — Appears to refuse a late event but malformed at the late leaves intent unclear.
Scorer: unacceptable — The draft is grammatically incorrect ('at the late') and fails to provide a useful counterproposal or a clear explanation for the conflict, making it an ineffective response.

**ONPOLICY-002-S4** · Context ['I need to be home by 21:00 tonight.'] · Incoming: Shall we start our two-hour movie at 20:00?
Draft: That works for me. Let's meet at the hotel at 5:00.
Assistant: uncertain — That works agrees to20:00 while5:00 could be a compatible earlier counterproposal; ambiguous commitment rather than definite violation.
Scorer: unacceptable — The draft proposes a meeting time of 5:00, which does not answer the peer's request to start a movie at 20:00 and ignores the user's constraint of needing to be home by 21:00.

**ONPOLICY-004-S3** · Context ['I can stay until 11:15, then I must leave.'] · Incoming: Could you stay until noon?
Draft: I cannot stay until eleven.
Assistant: uncertain — Declines staying until eleven rather than noon and appears inconsistent with can stay until11:15; uncertain intent.
Scorer: acceptable — The user explicitly stated they must leave at 11:15, and the draft correctly refuses the request to stay until noon.

**ONPOLICY-010-S3** · Context ['I borrowed your power bank for my trip.'] · Incoming: Could you bring my power bank back tomorrow?
Draft: I can't do it tomorrow. Could you use my laptop?
Assistant: uncertain — Explicit refusal is valid but laptop offer may be an irrelevant substitute for returning borrowed power bank.
Scorer: acceptable — The user explicitly refuses the request for tomorrow and offers a different action (lending a laptop), which is a valid counterproposal.

**ONPOLICY-004-S1** · Context ['I can stay until 11:15, then I must leave.'] · Incoming: Could you stay until noon?
Draft: Yes, tomorrow evening works for me.
Assistant: acceptable — Tomorrow evening is a plausible counterproposal, distinct from extending todays stay until noon.
Scorer: unacceptable — The draft provides a non-sequitur response; it ignores the request to stay until noon today and instead proposes a time for tomorrow evening without context.

**ONPOLICY-003-S1** · Context ['My appointment is at 16:00 and cannot be moved.'] · Incoming: Can you come for a meeting from 15:30 to 16:30?
Draft: I cannot join, can you reach in a few minutes?
Assistant: uncertain — Explicit refusal is valid, but can you reach in a few minutes is an unexplained follow-up; borderline coherence.
Scorer: acceptable — The user correctly refuses the requested time (15:30-16:30) because it conflicts with their 16:00 appointment and proposes a different action.

**ONPOLICY-006-S2** · Context ['I can attend the morning session, but I cannot attend the afternoon one.'] · Incoming: Please join the afternoon session with us.
Draft: I cannot attend it during the afternoon, but I cannot attend the afternoon session.
Assistant: acceptable — Repetitive but correctly refuses afternoon session; no role or constraint error.
Scorer: unacceptable — The draft is internally contradictory and nonsensical, repeating the same refusal twice in a single sentence.

**ONPOLICY-001-S4** · Context ['I cannot meet before 18:45 tonight.'] · Incoming: Would 17:30 work for dinner?
Draft: Let's meet at the mrt at the place.
Assistant: uncertain — Vague meeting proposal without resolving the conflicting requested time; no definite commitment to an hour.
Scorer: unacceptable — The draft fails to address the specific time request and ignores the user's stated constraint of not being able to meet before 18:45.

**ONPOLICY-003-S4** · Context ['My appointment is at 16:00 and cannot be moved.'] · Incoming: Can you come for a meeting from 15:30 to 16:30?
Draft: Sorry, I cannot come for the evening. Could we meet after that?
Assistant: uncertain — Could propose a later meeting, but misstates the request as evening and after that has an unclear referent.
Scorer: unacceptable — The user's draft claims they cannot come for the 'evening', but the peer's request is for a 15:30-16:30 meeting (afternoon). Furthermore, the draft fails to address the specific conflict with the 16:00 appointment mentioned in the history.

**ONPOLICY-002-S2** · Context ['I need to be home by 21:00 tonight.'] · Incoming: Shall we start our two-hour movie at 20:00?
Draft: I cannot make it tonight. Could we meet at the two-hour movie tomorrow then?
Assistant: acceptable — Declines tonight and offers tomorrow, a valid alternative.
Scorer: unacceptable — The draft contains an internal contradiction/logical error by suggesting to 'meet at the movie' as a time/location for a meeting, and it fails to provide a specific time for the proposed 'tomorrow' meeting.

**ONPOLICY-004-S4** · Context ['I can stay until 11:15, then I must leave.'] · Incoming: Could you stay until noon?
Draft: I cannot stay until six. Could we meet at the gate on Sunday?
Assistant: uncertain — Sunday proposal may be valid but six is unrelated to the noon request; ambiguous partial answer.
Scorer: unacceptable — The draft contains an internal contradiction and a non-sequitur; it mentions 'six' (an undefined time) while the conversation is about 'noon', and it introduces a new meeting location/day without addressing the specific request to stay until noon.

**ONPOLICY-011-S3** · Context ['I have your library card in my wallet.'] · Incoming: Do you still have my library card?
Draft: I have it, but I have to show you your library card in my wallet.
Assistant: acceptable — Confirms possession and offers to show peer their card; awkward but preserves roles and facts.
Scorer: unacceptable — The draft contains an internal contradiction/logical error: it states the user 'has' the card but then says they 'have to show' it in the wallet, which is a confusing way to confirm possession. More importantly, the draft's phrasing 'I have to show you your library card in my wallet' implies a future obligation or a physical impossibility of showing it while it is inside the wallet, making the commitment unclear.


## Final decision

Sampling354827 completed exit0 in17s; scoring354828 completed exit0 in5m17s. All96 calls completed, no malformed outputs. Same-case regression positive acceptance improved9/14 to10/14, negative rejection stayed23/23; one repaired definite case (ONPOLICY-006-S1), but repeated-refusal and tomorrow-evening false rejections persist. This is prompt-development evidence only.

Fresh audit gate FAILS:5/11 assistant-acceptable replies accepted,23/24 definite negatives rejected; balanced agreement70.64%. A material false positive ONPOLICY2-012-S2: context says peer borrowed USER cable, draft says USER will bring YOUR cable back; scorer wrongly calls USER borrower. Six assistant-acceptable replies rejected, including plausible next-Friday and tomorrow counterproposals and return-during-class. No post-score relabeling or threshold relaxation. Thirteen fresh assistant-uncertain cases are not counted as definite positives/negatives; several are awkward and subject to human disagreement. All fresh disagreements inspected in context. These are small correlated assistant-labeled samples, not human accuracy. Fresh score cannot be treated as paired evidence that v2 is worse than v1 because v1 was not run on this same fresh batch.

No RL training: revised scorer still rewards a clear ownership reversal and penalizes compatible alternatives. Avoid repeated prompt tuning on this audit. A different next approach would be a narrow mechanically verifiable structured-action RL task with explicit constraints/owners and a renderer, while stating that success there does not establish unrestricted conversational reply quality. No such pilot implemented or submitted in this task. All prior models/sources/data preserved, default unchanged, Streamlit remains closed.
