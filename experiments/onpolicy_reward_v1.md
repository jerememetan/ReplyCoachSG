# Sampled policy reply reward audit

No model training or scorer prompt edits during this audit. Labels are assistant contextual judgments frozen before scorer outcomes, not human gold.

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

## All sampled replies

| Sample | Assistant verdict | Scorer verdict | Reply |
|---|---|---|---|
| ONPOLICY-001-S1 | unacceptable | unacceptable | Sorry, 5:30 works for me. |
| ONPOLICY-001-S2 | unacceptable | unacceptable | Yes, 6:30 works for me. |
| ONPOLICY-001-S3 | acceptable | acceptable | I cannot make it tonight, sorry. |
| ONPOLICY-001-S4 | uncertain | unacceptable | Let's meet at the mrt at the place. |
| ONPOLICY-002-S1 | unacceptable | unacceptable | I would love to, but I can't finish it by seven. |
| ONPOLICY-002-S2 | acceptable | uncertain | I cannot make it tonight. Could we meet at the two-hour movie tomorrow then? |
| ONPOLICY-002-S3 | uncertain | acceptable | Sorry, I cannot make it at the late. |
| ONPOLICY-002-S4 | uncertain | unacceptable | That works for me. Let's meet at the hotel at 5:00. |
| ONPOLICY-003-S1 | uncertain | unacceptable | I cannot join, can you reach in a few minutes? |
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
| ONPOLICY-006-S1 | acceptable | unacceptable | I cannot, I cannot go anywhere else. |
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
| ONPOLICY-008-S4 | uncertain | unacceptable | I cannot make it for lunch today. |
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

## Disagreements

### ONPOLICY-003-S1
Context: ['My appointment is at 16:00 and cannot be moved.']
Incoming: Can you come for a meeting from 15:30 to 16:30?
Draft: I cannot join, can you reach in a few minutes?
Assistant judgment: uncertain: Explicit refusal is valid, but can you reach in a few minutes is an unexplained follow-up; borderline coherence.
Scorer: unacceptable: The user's appointment at 16:00 is fixed and cannot be moved, making a meeting from 15:30 to 16:30 impossible. The draft suggests a meeting in 'a few minutes' which does not resolve the conflict or provide a valid alternative.

### ONPOLICY-003-S4
Context: ['My appointment is at 16:00 and cannot be moved.']
Incoming: Can you come for a meeting from 15:30 to 16:30?
Draft: Sorry, I cannot come for the evening. Could we meet after that?
Assistant judgment: uncertain: Could propose a later meeting, but misstates the request as evening and after that has an unclear referent.
Scorer: unacceptable: The user's appointment is at 16:00 and cannot be moved, making the 15:30-16:30 slot impossible. However, the draft says 'evening' which is vague, and more importantly, the user's constraint is a specific time block. The draft is acceptable as a counterproposal, but the logic of 'cannot come for the evening' is slightly imprecise given the 16:00 constraint.

### ONPOLICY-002-S4
Context: ['I need to be home by 21:00 tonight.']
Incoming: Shall we start our two-hour movie at 20:00?
Draft: That works for me. Let's meet at the hotel at 5:00.
Assistant judgment: uncertain: That works agrees to20:00 while5:00 could be a compatible earlier counterproposal; ambiguous commitment rather than definite violation.
Scorer: unacceptable: The proposed start time of 20:00 for a two-hour movie means it ends at 22:00, which violates the user's constraint to be home by 21:00.

### ONPOLICY-002-S3
Context: ['I need to be home by 21:00 tonight.']
Incoming: Shall we start our two-hour movie at 20:00?
Draft: Sorry, I cannot make it at the late.
Assistant judgment: uncertain: Appears to refuse a late event but malformed at the late leaves intent unclear.
Scorer: acceptable: The user correctly identifies that a 20:00 start for a two-hour movie would end at 22:00, which violates their constraint of being home by 21:00.

### ONPOLICY-006-S2
Context: ['I can attend the morning session, but I cannot attend the afternoon one.']
Incoming: Please join the afternoon session with us.
Draft: I cannot attend it during the afternoon, but I cannot attend the afternoon session.
Assistant judgment: acceptable: Repetitive but correctly refuses afternoon session; no role or constraint error.
Scorer: unacceptable: The draft contains an internal contradiction and is logically incoherent.

### ONPOLICY-006-S1
Context: ['I can attend the morning session, but I cannot attend the afternoon one.']
Incoming: Please join the afternoon session with us.
Draft: I cannot, I cannot go anywhere else.
Assistant judgment: acceptable: Explicitly declines the afternoon invitation. Additional inability claim does not contradict morning attendance.
Scorer: unacceptable: The draft contradicts the user's initial statement that they can attend the morning session, and the phrase 'cannot go anywhere else' is an internal contradiction to the possibility of attending the morning session.

### ONPOLICY-004-S4
Context: ['I can stay until 11:15, then I must leave.']
Incoming: Could you stay until noon?
Draft: I cannot stay until six. Could we meet at the gate on Sunday?
Assistant judgment: uncertain: Sunday proposal may be valid but six is unrelated to the noon request; ambiguous partial answer.
Scorer: unacceptable: The user explicitly stated they must leave at 11:15, making a request for noon impossible. Additionally, the draft introduces a new time (six) and a new day (Sunday) without context.

### ONPOLICY-008-S4
Context: ['I am not free this weekend. I am free next Tuesday.']
Incoming: Shall we have lunch this Sunday?
Draft: I cannot make it for lunch today.
Assistant judgment: uncertain: Declines lunch today instead of Sunday; context does not establish today is Sunday.
Scorer: unacceptable: The user explicitly stated they are not free this weekend, and the draft fails to address the proposed Sunday lunch or the user's stated availability for next Tuesday.

### ONPOLICY-005-S1
Context: ['I am unavailable on Wednesday. Thursday evening is free.']
Incoming: Can we meet on Wednesday evening?
Draft: Let's meet at the same time as Wednesday evening.
Assistant judgment: uncertain: Same time as Wednesday evening could mean Wednesday itself or another day at the same clock time; cannot award a definite positive.
Scorer: unacceptable: The user explicitly stated they are unavailable on Wednesday.

### ONPOLICY-004-S1
Context: ['I can stay until 11:15, then I must leave.']
Incoming: Could you stay until noon?
Draft: Yes, tomorrow evening works for me.
Assistant judgment: acceptable: Tomorrow evening is a plausible counterproposal, distinct from extending todays stay until noon.
Scorer: unacceptable: The draft accepts a request to stay until noon despite the user explicitly stating they must leave at 11:15.

### ONPOLICY-011-S3
Context: ['I have your library card in my wallet.']
Incoming: Do you still have my library card?
Draft: I have it, but I have to show you your library card in my wallet.
Assistant judgment: acceptable: Confirms possession and offers to show peer their card; awkward but preserves roles and facts.
Scorer: unacceptable: The draft contains an internal contradiction and illogical phrasing regarding the action of showing the card.

### ONPOLICY-010-S3
Context: ['I borrowed your power bank for my trip.']
Incoming: Could you bring my power bank back tomorrow?
Draft: I can't do it tomorrow. Could you use my laptop?
Assistant judgment: uncertain: Explicit refusal is valid but laptop offer may be an irrelevant substitute for returning borrowed power bank.
Scorer: acceptable: The user correctly refuses the request for tomorrow and offers an alternative, which is a valid counterproposal.

### ONPOLICY-004-S3
Context: ['I can stay until 11:15, then I must leave.']
Incoming: Could you stay until noon?
Draft: I cannot stay until eleven.
Assistant judgment: uncertain: Declines staying until eleven rather than noon and appears inconsistent with can stay until11:15; uncertain intent.
Scorer: acceptable: The user is correctly refusing the request to stay until noon based on their stated constraint of leaving at 11:15.

### ONPOLICY-011-S2
Context: ['I have your library card in my wallet.']
Incoming: Do you still have my library card?
Draft: I have it now, but I have it last night.
Assistant judgment: uncertain: Affirms possession but last-night clause is malformed and contrast unclear; do not conflate tense error with definite role reversal.
Scorer: unacceptable: The draft contains an internal contradiction regarding the current state of possession.

### ONPOLICY-001-S4
Context: ['I cannot meet before 18:45 tonight.']
Incoming: Would 17:30 work for dinner?
Draft: Let's meet at the mrt at the place.
Assistant judgment: uncertain: Vague meeting proposal without resolving the conflicting requested time; no definite commitment to an hour.
Scorer: unacceptable: The proposed time (implied by the draft's acceptance of the meeting) violates the user's explicit constraint of not meeting before 18:45.

### ONPOLICY-002-S2
Context: ['I need to be home by 21:00 tonight.']
Incoming: Shall we start our two-hour movie at 20:00?
Draft: I cannot make it tonight. Could we meet at the two-hour movie tomorrow then?
Assistant judgment: acceptable: Declines tonight and offers tomorrow, a valid alternative.
Scorer: uncertain: Inconsistent verdict/issues


## Decision after full inspection

Sampling354821 and scoring354822 both completed exit0. No prompt or label edits after scoring. Frozen gate FAILS: positive acceptance9/14 (64.3%), negative rejection23/23 (100%), balanced agreement82.1%; one malformed verdict/issues combination preserved as uncertain/null with raw evidence. Four assistant-acceptable replies received negative scores; one received abstention. Nine of12 groups contained both positive and negative scores, so reward contrasts exist but are not sufficiently aligned with contextual adequacy. Eleven assistant-uncertain outputs were mostly assigned definite labels; they were not silently counted as known negatives or positives. All48 replies reviewed in context, including every disagreement.

Material scorer reasoning errors: calls repeated afternoon refusal an internal contradiction; rejects tomorrow-evening counterproposal as acceptance of today's noon; treats confirmation/showing possession as contradictory. Some cases are stylistically debatable and labels remain assistant judgments; the failures do not establish human accuracy or invalidate all scorer positives. Nonetheless current evidence fails the predefined pilot criterion. Do not run RL against these rewards or present diagnostic template pass as broad readiness. No actor weights updated and no deployment changes.

Next work should preserve these failures as development evidence, improve discrimination of refusal/counterproposal/repetition versus contradiction, and validate on a genuinely new sampled batch. Alternatively constrain a narrow pilot to mechanically verifiable structured actions, explicitly distinguishing that restricted task from natural-language reply quality. No fresh gate-passing claim or RL job submitted.
