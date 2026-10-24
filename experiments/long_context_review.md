# Long-context retrieval diagnostic review

> **Scope:** Engineering diagnostics only. These 20 draft fixtures are not an official benchmark.
> **Scores:** Retriever scores are not probabilities that a reply is correct.
> **Review:** Contextual-fit judgements require human review. No automatic quality winner is assigned.

## Configuration overview

| Retriever | Memory mode | Cases | Mean latency (ms) |
|---|---:|---:|---:|
| BM25 | M0 | 20 | 4.88 |
| BM25 | M1 | 20 | 9.03 |
| BM25 | M2 | 20 | 10.70 |
| MINILM | M0 | 20 | 25.65 |
| MINILM | M1 | 20 | 36.52 |
| MINILM | M2 | 20 | 43.31 |

## Case-by-case review

For each case, assess whether the retrieved candidates fit both the incoming message and the conversation. Do not judge by score alone.

Each case contains six configurations: BM25 and MiniLM, each with memory modes M0, M1, and M2. Candidate ranks and scores are preserved from the experiment outputs.

### DIAG-LC-001

- **Scenario:** `meals_and_food`
- **Intent:** `confirm_plan`
- **Incoming message:** peer: can we make it 7 instead?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The dinner plan; the proposed time changed to 7.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.650 | LOCAL | 7 lor | `study_sessions_and_spaces` | `request_information_or_status` |
| 2 | 0.457 | LOCAL | ok lor we eat at 2 | `meals_and_food` | `change_or_reschedule` |
| 3 | 0.423 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |

#### BM25 M1

**Selected memory**

- user: Okay, let me know when you're done.
- peer: The library is crowded today.
- user: We can find somewhere else.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.614 | LOCAL | can, lets find somewhere quiet in the library later | `study_sessions_and_spaces` | `invite_or_propose` |
| 1 | 0.600 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.524 | LOCAL | Okay, thanks. I'll find other place. | `study_sessions_and_spaces` | `inform_or_update` |

#### BM25 M2

**Selected memory**

- peer: Want to get dinner after class?
- peer: I have a meeting about the project first.
- peer: The library is crowded today.
- user: We can find somewhere else.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.645 | LOCAL | can, lets find somewhere quiet in the library later | `study_sessions_and_spaces` | `invite_or_propose` |
| 1 | 0.624 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.616 | LOCAL | Can, we discuss it after class bah. | `assignments_and_group_projects` | `invite_or_propose` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.285 | LOCAL | Ya seven can | `meals_and_food` | `change_or_reschedule` |
| 1 | 0.283 | LOCAL | Can, eight works for me. | `study_sessions_and_spaces` | `change_or_reschedule` |
| 3 | 0.180 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |

#### MINILM M1

**Selected memory**

- user: Okay, let me know when you're done.
- peer: The library is crowded today.
- user: We can find somewhere else.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.294 | LOCAL | can, lets find somewhere quiet in the library later | `study_sessions_and_spaces` | `invite_or_propose` |
| 1 | 0.241 | LOCAL | ya, 2 still can. see you thr | `study_sessions_and_spaces` | `confirm_plan` |
| 3 | 0.217 | LOCAL | ya i cmg | `study_sessions_and_spaces` | `remind_or_nudge` |

#### MINILM M2

**Selected memory**

- peer: Want to get dinner after class?
- peer: I have a meeting about the project first.
- peer: The library is crowded today.
- user: We can find somewhere else.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.250 | LOCAL | Ya, three still can. | `study_sessions_and_spaces` | `confirm_plan` |
| 2 | 0.249 | LOCAL | Can, two works for me. | `classes_and_tutorials` | `change_or_reschedule` |
| 3 | 0.247 | LOCAL | Can, eight works for me. | `study_sessions_and_spaces` | `change_or_reschedule` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-002

- **Scenario:** `study_sessions_and_spaces`
- **Intent:** `change_or_reschedule`
- **Incoming message:** peer: library too packed, can switch?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The study meetup; the location may need to change.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.636 | LOCAL | Ya hor, let's ask them too. | `study_sessions_and_spaces` | `invite_or_propose` |
| 3 | 0.441 | LOCAL | no the library full liao | `study_sessions_and_spaces` | `request_information_or_status` |
| 2 | 0.435 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |

#### BM25 M1

**Selected memory**

- user: Okay.
- peer: I need to print my notes too.
- user: There is a printer near the lab.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.615 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.505 | LOCAL | Eh, can help me print also? | `campus_services_and_equipment` | `invite_or_propose` |
| 3 | 0.461 | LOCAL | can, we compare them after class | `course_materials` | `invite_or_propose` |

#### BM25 M2

**Selected memory**

- peer: Let's study together tomorrow.
- peer: The main library is my first choice.
- peer: I need to print my notes too.
- user: There is a printer near the lab.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.613 | LOCAL | Ya hor, let's ask them too. | `study_sessions_and_spaces` | `invite_or_propose` |
| 1 | 0.600 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |
| 3 | 0.581 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.220 | LOCAL | no the library full liao | `study_sessions_and_spaces` | `request_information_or_status` |
| 1 | 0.182 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |
| 3 | 0.178 | LOCAL | ya i cmg | `study_sessions_and_spaces` | `remind_or_nudge` |

#### MINILM M1

**Selected memory**

- user: Okay.
- peer: I need to print my notes too.
- user: There is a printer near the lab.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.327 | LOCAL | can i pay u ltr | `campus_services_and_equipment` | `inform_or_update` |
| 2 | 0.283 | LOCAL | ya, damn long, got like 4 other ppl before us | `campus_services_and_equipment` | `request_information_or_status` |
| 3 | 0.281 | LOCAL | ok dats fine | `campus_services_and_equipment` | `decline` |

#### MINILM M2

**Selected memory**

- peer: Let's study together tomorrow.
- peer: The main library is my first choice.
- peer: I need to print my notes too.
- user: There is a printer near the lab.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.287 | LOCAL | can, lets find somewhere quiet in the library later | `study_sessions_and_spaces` | `invite_or_propose` |
| 3 | 0.256 | LOCAL | can i pay u ltr | `campus_services_and_equipment` | `inform_or_update` |
| 2 | 0.238 | LOCAL | Okay, thanks. I'll find other place. | `study_sessions_and_spaces` | `inform_or_update` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-003

- **Scenario:** `course_materials`
- **Intent:** `request_help_or_resource`
- **Incoming message:** peer: can send those over tonight?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The user’s lecture notes, not the quiz announcement.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.700 | LOCAL | Yea, I send tonight. | `course_materials` | `confirm_plan` |
| 3 | 0.426 | LOCAL | okay, i come over soon then | `social_events_and_outings` | `inform_or_update` |
| 2 | 0.426 | LOCAL | Paiseh, now cannot. I'll send them later. | `course_materials` | `decline` |

#### BM25 M1

**Selected memory**

- user: I'll tidy them up later.
- peer: Do you know when the quiz is?
- user: I think the announcement is on the course page.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.620 | LOCAL | Paiseh, now cannot. I'll send them later. | `course_materials` | `decline` |
| 1 | 0.600 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.513 | LOCAL | can ah, we can do tgt tomorrow night | `course_materials` | `invite_or_propose` |

#### BM25 M2

**Selected memory**

- peer: I missed the first half of lecture.
- peer: Thanks, that would help.
- peer: Do you know when the quiz is?
- user: I think the announcement is on the course page.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.648 | LOCAL | Ya, the slides are on the course page already. | `course_materials` | `request_information_or_status` |
| 2 | 0.473 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.455 | LOCAL | okay! I order noodles for you | `meals_and_food` | `inform_or_update` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.371 | LOCAL | Yea, I send tonight. | `course_materials` | `confirm_plan` |
| 3 | 0.300 | LOCAL | can, i send you the photo now | `course_materials` | `request_help_or_resource` |
| 2 | 0.270 | LOCAL | Paiseh, now cannot. I'll send them later. | `course_materials` | `decline` |

#### MINILM M1

**Selected memory**

- user: I'll tidy them up later.
- peer: Do you know when the quiz is?
- user: I think the announcement is on the course page.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.296 | LOCAL | Paiseh, now cannot. I'll send them later. | `course_materials` | `decline` |
| 1 | 0.282 | LOCAL | can ah, we can do tgt tomorrow night | `course_materials` | `invite_or_propose` |
| 3 | 0.270 | LOCAL | Ya, the slides are on the course page already. | `course_materials` | `request_information_or_status` |

#### MINILM M2

**Selected memory**

- peer: I missed the first half of lecture.
- peer: Thanks, that would help.
- peer: Do you know when the quiz is?
- user: I think the announcement is on the course page.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.339 | LOCAL | Ya, the slides are on the course page already. | `course_materials` | `request_information_or_status` |
| 2 | 0.293 | LOCAL | Paiseh, now cannot. I'll send them later. | `course_materials` | `decline` |
| 3 | 0.273 | LOCAL | at seminar room. why eh | `classes_and_tutorials` | `request_information_or_status` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-004

- **Scenario:** `assignments_and_group_projects`
- **Intent:** `confirm_plan`
- **Incoming message:** peer: you still doing the analysis right?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The user's assigned project section is the analysis.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.667 | LOCAL | ya, 2 still can. see you thr | `study_sessions_and_spaces` | `confirm_plan` |
| 2 | 0.588 | LOCAL | ya, still on. we makan after class | `meals_and_food` | `confirm_plan` |
| 3 | 0.565 | LOCAL | uhh, i think still need do slides | `course_materials` | `remind_or_nudge` |

#### BM25 M1

**Selected memory**

- user: Okay.
- peer: The presentation room has changed.
- user: I'll check the new room.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.741 | LOCAL | Okay, thanks. I'll check the new room. | `classes_and_tutorials` | `inform_or_update` |
| 2 | 0.535 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.494 | LOCAL | okay, i come level 4 now | `classes_and_tutorials` | `inform_or_update` |

#### BM25 M2

**Selected memory**

- peer: I'll do the literature review.
- user: I'll handle the analysis.
- peer: The presentation room has changed.
- user: I'll check the new room.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.695 | LOCAL | Okay, thanks. I'll check the new room. | `classes_and_tutorials` | `inform_or_update` |
| 2 | 0.541 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.466 | LOCAL | okay, i come level 4 now | `classes_and_tutorials` | `inform_or_update` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.227 | LOCAL | I steady one, I checked the information. | `course_materials` | `confirm_plan` |
| 3 | 0.224 | LOCAL | uhh, i think still need do slides | `course_materials` | `remind_or_nudge` |
| 1 | 0.210 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |

#### MINILM M1

**Selected memory**

- user: Okay.
- peer: The presentation room has changed.
- user: I'll check the new room.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.377 | LOCAL | Okay, thanks. I'll check the new room. | `classes_and_tutorials` | `inform_or_update` |
| 2 | 0.235 | LOCAL | uhh, i think still need do slides | `course_materials` | `remind_or_nudge` |
| 3 | 0.212 | LOCAL | oh ok nvm all g | `assignments_and_group_projects` | `decline` |

#### MINILM M2

**Selected memory**

- peer: I'll do the literature review.
- user: I'll handle the analysis.
- peer: The presentation room has changed.
- user: I'll check the new room.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 3 | 0.225 | LOCAL | uhh, i think still need do slides | `course_materials` | `remind_or_nudge` |
| 1 | 0.222 | LOCAL | I will one, dont worry | `assignments_and_group_projects` | `remind_or_nudge` |
| 2 | 0.211 | LOCAL | Ya hor, let's ask them too. | `study_sessions_and_spaces` | `invite_or_propose` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-005

- **Scenario:** `meetups_and_transport`
- **Intent:** `inform_or_update`
- **Incoming message:** peer: I'm here already, where are you?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The meetup location is the MRT Exit B.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 3 | 0.632 | LOCAL | you can't, it got choped already | `assignments_and_group_projects` | `request_information_or_status` |
| 1 | 0.600 | LOCAL | OK, I come now | `meetups_and_transport` | `request_help_or_resource` |
| 2 | 0.592 | LOCAL | Nevermind, just update me when you reach. | `meetups_and_transport` | `inform_or_update` |

#### BM25 M1

**Selected memory**

- user: I'll walk over.
- peer: The bus is slow today.
- user: No worries, update me when you arrive.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.664 | LOCAL | Nevermind, just update me when you reach. | `meetups_and_transport` | `inform_or_update` |
| 2 | 0.301 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.282 | LOCAL | No, I taking bus, MRT very slow | `meetups_and_transport` | `request_information_or_status` |

#### BM25 M2

**Selected memory**

- peer: Let's meet at the MRT after class.
- peer: I'll wait near Exit B.
- peer: The bus is slow today.
- user: No worries, update me when you arrive.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.648 | LOCAL | Nevermind, just update me when you reach. | `meetups_and_transport` | `inform_or_update` |
| 2 | 0.380 | LOCAL | it's okay lah, i wait here. come when you can | `sports_and_exercise` | `apologise_or_explain` |
| 3 | 0.309 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 3 | 0.248 | LOCAL | havent leh, i was waiting for you | `meals_and_food` | `request_information_or_status` |
| 1 | 0.226 | LOCAL | OK, I come now | `meetups_and_transport` | `request_help_or_resource` |
| 2 | 0.218 | LOCAL | Ok I coming liao. | `meals_and_food` | `inform_or_update` |

#### MINILM M1

**Selected memory**

- user: I'll walk over.
- peer: The bus is slow today.
- user: No worries, update me when you arrive.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.347 | LOCAL | Nevermind, just update me when you reach. | `meetups_and_transport` | `inform_or_update` |
| 1 | 0.304 | LOCAL | kk , take ur time ah, i walk ard first. | `meetups_and_transport` | `apologise_or_explain` |
| 3 | 0.237 | LOCAL | Ok I coming liao. | `meals_and_food` | `inform_or_update` |

#### MINILM M2

**Selected memory**

- peer: Let's meet at the MRT after class.
- peer: I'll wait near Exit B.
- peer: The bus is slow today.
- user: No worries, update me when you arrive.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.361 | LOCAL | No, I taking bus, MRT very slow | `meetups_and_transport` | `request_information_or_status` |
| 1 | 0.319 | LOCAL | kk u go first | `meetups_and_transport` | `decline` |
| 3 | 0.311 | LOCAL | sorry ah i reaching in 5 mins, got train delay | `assignments_and_group_projects` | `request_information_or_status` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-006

- **Scenario:** `campus_services_and_equipment`
- **Intent:** `request_help_or_resource`
- **Incoming message:** peer: can bring that when we meet?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The charger is the item to bring, not the assignment.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | ok, another day bah | `assignments_and_group_projects` | `change_or_reschedule` |
| 2 | 0.554 | LOCAL | Abit far leh, but okay la | `study_sessions_and_spaces` | `apologise_or_explain` |
| 3 | 0.517 | LOCAL | eh paiseh, i forgot | `study_sessions_and_spaces` | `apologise_or_explain` |

#### BM25 M1

**Selected memory**

- user: Sure.
- peer: I need to print my assignment too.
- user: The printer is on the second floor.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.667 | LOCAL | Can try the second floor one. | `campus_services_and_equipment` | `request_information_or_status` |
| 2 | 0.488 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.351 | LOCAL | Paiseh, can't join today. Need to finish my assignment. | `sports_and_exercise` | `decline` |

#### BM25 M2

**Selected memory**

- peer: My laptop is nearly out of battery.
- peer: Could I borrow it later?
- peer: I need to print my assignment too.
- user: The printer is on the second floor.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.650 | LOCAL | Can try the second floor one. | `campus_services_and_equipment` | `request_information_or_status` |
| 2 | 0.438 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.420 | LOCAL | okay! I order noodles for you | `meals_and_food` | `inform_or_update` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.285 | LOCAL | can, mrt easier. i meet you there | `meetups_and_transport` | `change_or_reschedule` |
| 2 | 0.243 | LOCAL | Can, two works for me. | `classes_and_tutorials` | `change_or_reschedule` |
| 3 | 0.198 | LOCAL | eh paiseh, i forgot | `study_sessions_and_spaces` | `apologise_or_explain` |

#### MINILM M1

**Selected memory**

- user: Sure.
- peer: I need to print my assignment too.
- user: The printer is on the second floor.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.415 | LOCAL | Can try the second floor one. | `campus_services_and_equipment` | `request_information_or_status` |
| 3 | 0.298 | LOCAL | Eh, can help me print also? | `campus_services_and_equipment` | `invite_or_propose` |
| 2 | 0.295 | LOCAL | press yellow button first can already | `campus_services_and_equipment` | `request_information_or_status` |

#### MINILM M2

**Selected memory**

- peer: My laptop is nearly out of battery.
- peer: Could I borrow it later?
- peer: I need to print my assignment too.
- user: The printer is on the second floor.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.310 | LOCAL | Can try the second floor one. | `campus_services_and_equipment` | `request_information_or_status` |
| 3 | 0.244 | LOCAL | can i pay u ltr | `campus_services_and_equipment` | `inform_or_update` |
| 2 | 0.221 | LOCAL | ya, damn long, got like 4 other ppl before us | `campus_services_and_equipment` | `request_information_or_status` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-007

- **Scenario:** `sports_and_exercise`
- **Intent:** `change_or_reschedule`
- **Incoming message:** peer: court only available at 7 now
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The badminton booking time changed to 7.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.650 | LOCAL | 7 lor | `study_sessions_and_spaces` | `request_information_or_status` |
| 3 | 0.526 | LOCAL | can lah, small table also okay. i come over now | `meals_and_food` | `inform_or_update` |
| 2 | 0.517 | LOCAL | aiyo, i go see if got other courts | `sports_and_exercise` | `inform_or_update` |

#### BM25 M1

**Selected memory**

- user: Let me know the timing.
- peer: I have a lecture in the morning.
- user: We can play later.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.614 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.493 | LOCAL | Can,  I book later. | `sports_and_exercise` | `request_help_or_resource` |
| 3 | 0.375 | LOCAL | Paiseh, now cannot. I'll send them later. | `course_materials` | `decline` |

#### BM25 M2

**Selected memory**

- peer: Want to play badminton Friday?
- peer: I'll try to book a court.
- peer: I have a lecture in the morning.
- user: We can play later.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.626 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.561 | LOCAL | Can,  I book later. | `sports_and_exercise` | `request_help_or_resource` |
| 3 | 0.488 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.334 | LOCAL | aiyo, i go see if got other courts | `sports_and_exercise` | `inform_or_update` |
| 2 | 0.283 | LOCAL | Can,  I book later. | `sports_and_exercise` | `request_help_or_resource` |
| 3 | 0.233 | LOCAL | it's okay lah, i wait here. come when you can | `sports_and_exercise` | `apologise_or_explain` |

#### MINILM M1

**Selected memory**

- user: Let me know the timing.
- peer: I have a lecture in the morning.
- user: We can play later.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.314 | LOCAL | Can,  I book later. | `sports_and_exercise` | `request_help_or_resource` |
| 1 | 0.294 | LOCAL | it's okay lah, i wait here. come when you can | `sports_and_exercise` | `apologise_or_explain` |
| 3 | 0.265 | LOCAL | can, 5:30 no problem. you come after class | `assignments_and_group_projects` | `change_or_reschedule` |

#### MINILM M2

**Selected memory**

- peer: Want to play badminton Friday?
- peer: I'll try to book a court.
- peer: I have a lecture in the morning.
- user: We can play later.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.356 | LOCAL | it's okay lah, i wait here. come when you can | `sports_and_exercise` | `apologise_or_explain` |
| 3 | 0.344 | LOCAL | okok, i book later | `sports_and_exercise` | `remind_or_nudge` |
| 2 | 0.338 | LOCAL | can ah | `sports_and_exercise` | `request_help_or_resource` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-008

- **Scenario:** `social_events_and_outings`
- **Intent:** `confirm_plan`
- **Incoming message:** peer: still on for Saturday?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The exhibition visit on Saturday.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.867 | LOCAL | Yeah, still on for Saturday. | `social_events_and_outings` | `confirm_plan` |
| 3 | 0.269 | LOCAL | uhh saturday sunday can? | `meals_and_food` | `change_or_reschedule` |
| 2 | 0.221 | LOCAL | ok, another day bah | `assignments_and_group_projects` | `change_or_reschedule` |

#### BM25 M1

**Selected memory**

- user: Okay.
- peer: The cafe nearby looks nice.
- user: We can eat after.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.571 | LOCAL | Yeah, still on for Saturday. | `social_events_and_outings` | `confirm_plan` |
| 3 | 0.488 | LOCAL | ya, still on. we makan after class | `meals_and_food` | `confirm_plan` |

#### BM25 M2

**Selected memory**

- peer: Want to go to the exhibition Saturday?
- peer: I'll check the tickets.
- peer: The cafe nearby looks nice.
- user: We can eat after.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.670 | LOCAL | Yeah, still on for Saturday. | `social_events_and_outings` | `confirm_plan` |
| 2 | 0.527 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.472 | LOCAL | idw, i cannot eat spicy | `meals_and_food` | `decline` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.545 | LOCAL | Yeah, still on for Saturday. | `social_events_and_outings` | `confirm_plan` |
| 2 | 0.279 | LOCAL | uhh saturday sunday can? | `meals_and_food` | `change_or_reschedule` |
| 3 | 0.205 | LOCAL | ya, tuesday no class | `classes_and_tutorials` | `request_information_or_status` |

#### MINILM M1

**Selected memory**

- user: Okay.
- peer: The cafe nearby looks nice.
- user: We can eat after.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.295 | LOCAL | ok lor we eat at 2 | `meals_and_food` | `change_or_reschedule` |
| 1 | 0.265 | LOCAL | walao | `meals_and_food` | `change_or_reschedule` |
| 3 | 0.246 | LOCAL | Paiseh, today cannot. Need finish some work. | `meals_and_food` | `decline` |

#### MINILM M2

**Selected memory**

- peer: Want to go to the exhibition Saturday?
- peer: I'll check the tickets.
- peer: The cafe nearby looks nice.
- user: We can eat after.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.367 | LOCAL | Yeah, still on for Saturday. | `social_events_and_outings` | `confirm_plan` |
| 3 | 0.279 | LOCAL | Thanks for asking, but tonight cannot ah | `social_events_and_outings` | `decline` |
| 2 | 0.266 | LOCAL | of course got, I steady one | `social_events_and_outings` | `remind_or_nudge` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-009

- **Scenario:** `classes_and_tutorials`
- **Intent:** `request_information_or_status`
- **Incoming message:** peer: which room was it again?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The tutorial room, which must be checked rather than invented.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.657 | LOCAL | which movie? | `social_events_and_outings` | `invite_or_propose` |
| 2 | 0.643 | LOCAL | my bad. I thought it was ok | `course_materials` | `apologise_or_explain` |
| 3 | 0.591 | LOCAL | u want sit which row | `classes_and_tutorials` | `request_help_or_resource` |

#### BM25 M1

**Selected memory**

- user: I'll let you know.
- peer: Did you finish the worksheet?
- user: Almost.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.437 | LOCAL | Okay, thanks. I'll check the new room. | `classes_and_tutorials` | `inform_or_update` |
| 3 | 0.398 | LOCAL | gna finish soon. whats ur ans for the first question | `assignments_and_group_projects` | `request_information_or_status` |

#### BM25 M2

**Selected memory**

- peer: Do you know our tutorial room?
- peer: I may arrive a little late.
- peer: Did you finish the worksheet?
- user: Almost.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.629 | LOCAL | okay! I order noodles for you | `meals_and_food` | `inform_or_update` |
| 1 | 0.600 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.590 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.229 | LOCAL | okay, i come level 4 now | `classes_and_tutorials` | `inform_or_update` |
| 2 | 0.198 | LOCAL | okay lor, i check downstairs instead | `study_sessions_and_spaces` | `inform_or_update` |
| 3 | 0.197 | LOCAL | got, third floor. I saved one for you alr | `study_sessions_and_spaces` | `request_information_or_status` |

#### MINILM M1

**Selected memory**

- user: I'll let you know.
- peer: Did you finish the worksheet?
- user: Almost.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.223 | LOCAL | Paiseh, can't join today. Need to finish my assignment. | `sports_and_exercise` | `decline` |
| 1 | 0.212 | LOCAL | Okay la, understand | `course_materials` | `decline` |
| 3 | 0.202 | LOCAL | inside alr, but hasnt start yet | `classes_and_tutorials` | `request_information_or_status` |

#### MINILM M2

**Selected memory**

- peer: Do you know our tutorial room?
- peer: I may arrive a little late.
- peer: Did you finish the worksheet?
- user: Almost.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.250 | LOCAL | inside alr, but hasnt start yet | `classes_and_tutorials` | `request_information_or_status` |
| 2 | 0.248 | LOCAL | Sure, faster ah | `classes_and_tutorials` | `remind_or_nudge` |
| 3 | 0.242 | LOCAL | not yet leh, lecturer also not here | `classes_and_tutorials` | `request_information_or_status` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-010

- **Scenario:** `meals_and_food`
- **Intent:** `decline`
- **Incoming message:** peer: coming for lunch or not?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The user has a deadline and may need to decline.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | i anything | `meals_and_food` | `request_information_or_status` |
| 2 | 0.538 | LOCAL | havent leh, i was waiting for you | `meals_and_food` | `request_information_or_status` |
| 3 | 0.445 | LOCAL | got, third floor. I saved one for you alr | `study_sessions_and_spaces` | `request_information_or_status` |

#### BM25 M1

**Selected memory**

- user: I have a deadline today.
- peer: The canteen is crowded.
- user: Maybe another day.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.616 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.391 | LOCAL | ok, another day bah | `assignments_and_group_projects` | `change_or_reschedule` |
| 2 | 0.360 | LOCAL | Okay, thanks. I'll find other place. | `study_sessions_and_spaces` | `inform_or_update` |

#### BM25 M2

**Selected memory**

- peer: Let's get lunch after class.
- peer: No worries.
- peer: The canteen is crowded.
- user: Maybe another day.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.564 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |
| 3 | 0.484 | LOCAL | idw, i cannot eat spicy | `meals_and_food` | `decline` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.273 | LOCAL | ok lor we eat at 2 | `meals_and_food` | `change_or_reschedule` |
| 2 | 0.244 | LOCAL | aiyo, oops | `meals_and_food` | `confirm_plan` |
| 3 | 0.239 | LOCAL | walao | `meals_and_food` | `change_or_reschedule` |

#### MINILM M1

**Selected memory**

- user: I have a deadline today.
- peer: The canteen is crowded.
- user: Maybe another day.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.299 | LOCAL | ok lor we eat at 2 | `meals_and_food` | `change_or_reschedule` |
| 2 | 0.271 | LOCAL | Paiseh, today cannot. Need finish some work. | `meals_and_food` | `decline` |
| 3 | 0.250 | LOCAL | eh your part finish alr anot | `assignments_and_group_projects` | `apologise_or_explain` |

#### MINILM M2

**Selected memory**

- peer: Let's get lunch after class.
- peer: No worries.
- peer: The canteen is crowded.
- user: Maybe another day.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.308 | LOCAL | ok lor we eat at 2 | `meals_and_food` | `change_or_reschedule` |
| 2 | 0.307 | LOCAL | ya, tuesday no class | `classes_and_tutorials` | `request_information_or_status` |
| 3 | 0.261 | LOCAL | Paiseh, today cannot. Need finish some work. | `meals_and_food` | `decline` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-011

- **Scenario:** `course_materials`
- **Intent:** `request_information_or_status`
- **Incoming message:** peer: found the recording too, want it?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The recording is a newly offered resource; do not confuse it with slides.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | Yea, I send tonight. | `course_materials` | `confirm_plan` |
| 2 | 0.560 | LOCAL | nice, found alr can. thanks | `course_materials` | `inform_or_update` |
| 3 | 0.523 | LOCAL | Can, I'll send you the link. | `course_materials` | `request_help_or_resource` |

#### BM25 M1

**Selected memory**

- user: I'll look later.
- peer: The tutorial was quite hard.
- user: Yeah, lots to cover.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.616 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.420 | LOCAL | Can, I'll send you the link. | `course_materials` | `request_help_or_resource` |
| 2 | 0.392 | LOCAL | Yea, I send tonight. | `course_materials` | `confirm_plan` |

#### BM25 M2

**Selected memory**

- peer: Did the professor upload the slides?
- peer: I need them to revise.
- peer: The tutorial was quite hard.
- user: Yeah, lots to cover.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.613 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.488 | LOCAL | idw, i cannot eat spicy | `meals_and_food` | `decline` |
| 2 | 0.483 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.221 | LOCAL | Can, I'll send you the link. | `course_materials` | `request_help_or_resource` |
| 1 | 0.213 | LOCAL | Yea, I send tonight. | `course_materials` | `confirm_plan` |
| 3 | 0.157 | LOCAL | no u gotta for it this shit aint free | `course_materials` | `invite_or_propose` |

#### MINILM M1

**Selected memory**

- user: I'll look later.
- peer: The tutorial was quite hard.
- user: Yeah, lots to cover.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.392 | LOCAL | Can, I'll send you the link. | `course_materials` | `request_help_or_resource` |
| 2 | 0.211 | LOCAL | 5  sessions left | `classes_and_tutorials` | `request_information_or_status` |
| 3 | 0.192 | LOCAL | eh awesome la thanks! | `classes_and_tutorials` | `invite_or_propose` |

#### MINILM M2

**Selected memory**

- peer: Did the professor upload the slides?
- peer: I need them to revise.
- peer: The tutorial was quite hard.
- user: Yeah, lots to cover.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.341 | LOCAL | nice, found alr can. thanks | `course_materials` | `inform_or_update` |
| 2 | 0.324 | LOCAL | Can, I'll send you the link. | `course_materials` | `request_help_or_resource` |
| 3 | 0.318 | LOCAL | got, i send you in a bit | `course_materials` | `request_information_or_status` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-012

- **Scenario:** `assignments_and_group_projects`
- **Intent:** `remind_or_nudge`
- **Incoming message:** peer: remember to upload yours soon
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The user's own project section should be uploaded before the deadline.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.627 | LOCAL | gna finish soon. whats ur ans for the first question | `assignments_and_group_projects` | `request_information_or_status` |
| 2 | 0.426 | LOCAL | Ya, the slides are on the course page already. | `course_materials` | `request_information_or_status` |
| 3 | 0.396 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |

#### BM25 M1

**Selected memory**

- user: Mine needs one more edit.
- peer: We submit before midnight.
- user: I'll finish it.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.616 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.287 | LOCAL | Deadline is Friday evening. | `assignments_and_group_projects` | `request_information_or_status` |
| 3 | 0.276 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |

#### BM25 M2

**Selected memory**

- peer: I'll upload my section tonight.
- peer: The references are almost done.
- peer: We submit before midnight.
- user: I'll finish it.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.614 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.525 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |
| 3 | 0.515 | LOCAL | Deadline is Friday evening. | `assignments_and_group_projects` | `request_information_or_status` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.198 | LOCAL | Yea, I send tonight. | `course_materials` | `confirm_plan` |
| 2 | 0.191 | LOCAL | ya, damn long, got like 4 other ppl before us | `campus_services_and_equipment` | `request_information_or_status` |
| 3 | 0.189 | LOCAL | Can, I'll send you the link. | `course_materials` | `request_help_or_resource` |

#### MINILM M1

**Selected memory**

- user: Mine needs one more edit.
- peer: We submit before midnight.
- user: I'll finish it.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.251 | LOCAL | eh your part finish alr anot | `assignments_and_group_projects` | `apologise_or_explain` |
| 2 | 0.224 | LOCAL | Deadline is Friday evening. | `assignments_and_group_projects` | `request_information_or_status` |
| 3 | 0.216 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |

#### MINILM M2

**Selected memory**

- peer: I'll upload my section tonight.
- peer: The references are almost done.
- peer: We submit before midnight.
- user: I'll finish it.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.301 | LOCAL | Deadline is Friday evening. | `assignments_and_group_projects` | `request_information_or_status` |
| 2 | 0.271 | LOCAL | eh your part finish alr anot | `assignments_and_group_projects` | `apologise_or_explain` |
| 3 | 0.245 | LOCAL | oh ok nvm all g | `assignments_and_group_projects` | `decline` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-013

- **Scenario:** `meetups_and_transport`
- **Intent:** `change_or_reschedule`
- **Incoming message:** peer: running late, can meet at 5.30?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The meetup time changes from 5 to 5.30.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.686 | LOCAL | can, 5:30 no problem. you come after class | `assignments_and_group_projects` | `change_or_reschedule` |
| 2 | 0.297 | LOCAL | kk u go first | `meetups_and_transport` | `decline` |
| 3 | 0.271 | LOCAL | Faster leh, professor going to start already. | `classes_and_tutorials` | `inform_or_update` |

#### BM25 M1

**Selected memory**

- user: I'll leave class first.
- peer: The canteen is noisy.
- user: We can talk somewhere else.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.662 | LOCAL | can, 5:30 no problem. you come after class | `assignments_and_group_projects` | `change_or_reschedule` |
| 2 | 0.552 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.368 | LOCAL | kk u go first | `meetups_and_transport` | `decline` |

#### BM25 M2

**Selected memory**

- peer: Meet me outside the lecture hall.
- peer: I'll be there at 5.
- peer: The canteen is noisy.
- user: We can talk somewhere else.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.640 | LOCAL | can, 5:30 no problem. you come after class | `assignments_and_group_projects` | `change_or_reschedule` |
| 3 | 0.503 | LOCAL | can, mrt easier. i meet you there | `meetups_and_transport` | `change_or_reschedule` |
| 2 | 0.485 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.394 | LOCAL | can, 5:30 no problem. you come after class | `assignments_and_group_projects` | `change_or_reschedule` |
| 3 | 0.320 | LOCAL | Ok, can | `meetups_and_transport` | `change_or_reschedule` |
| 2 | 0.307 | LOCAL | ya c u later | `sports_and_exercise` | `remind_or_nudge` |

#### MINILM M1

**Selected memory**

- user: I'll leave class first.
- peer: The canteen is noisy.
- user: We can talk somewhere else.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.275 | LOCAL | at seminar room. why eh | `classes_and_tutorials` | `request_information_or_status` |
| 2 | 0.250 | LOCAL | not yet leh, lecturer also not here | `classes_and_tutorials` | `request_information_or_status` |
| 3 | 0.230 | LOCAL | Paiseh, can't join today. Need to finish my assignment. | `sports_and_exercise` | `decline` |

#### MINILM M2

**Selected memory**

- peer: Meet me outside the lecture hall.
- peer: I'll be there at 5.
- peer: The canteen is noisy.
- user: We can talk somewhere else.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.352 | LOCAL | can, mrt easier. i meet you there | `meetups_and_transport` | `change_or_reschedule` |
| 3 | 0.313 | LOCAL | can, lets find somewhere quiet in the library later | `study_sessions_and_spaces` | `invite_or_propose` |
| 2 | 0.278 | LOCAL | ya c u later | `sports_and_exercise` | `remind_or_nudge` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-014

- **Scenario:** `campus_services_and_equipment`
- **Intent:** `request_help_or_resource`
- **Incoming message:** peer: which floor was the printer on?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The printer location is on the second floor.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.673 | LOCAL | Can try the second floor one. | `campus_services_and_equipment` | `request_information_or_status` |
| 2 | 0.404 | LOCAL | got, third floor. I saved one for you alr | `study_sessions_and_spaces` | `request_information_or_status` |
| 3 | 0.366 | LOCAL | which movie? | `social_events_and_outings` | `invite_or_propose` |

#### BM25 M1

**Selected memory**

- user: No problem.
- peer: I need to submit a report.
- user: Hope you can print it in time.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.646 | LOCAL | Can try the second floor one. | `campus_services_and_equipment` | `request_information_or_status` |
| 2 | 0.469 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.397 | LOCAL | wah then how u printing the cheatsheet | `campus_services_and_equipment` | `inform_or_update` |

#### BM25 M2

**Selected memory**

- user: There's a printer upstairs.
- peer: Thanks.
- peer: I need to submit a report.
- user: Hope you can print it in time.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.644 | LOCAL | Can try the second floor one. | `campus_services_and_equipment` | `request_information_or_status` |
| 2 | 0.464 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.305 | LOCAL | press yellow button first can already | `campus_services_and_equipment` | `request_information_or_status` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.430 | LOCAL | Can try the second floor one. | `campus_services_and_equipment` | `request_information_or_status` |
| 2 | 0.255 | LOCAL | aiyo, got any other printers nearby | `campus_services_and_equipment` | `inform_or_update` |
| 3 | 0.254 | LOCAL | ya, damn long, got like 4 other ppl before us | `campus_services_and_equipment` | `request_information_or_status` |

#### MINILM M1

**Selected memory**

- user: No problem.
- peer: I need to submit a report.
- user: Hope you can print it in time.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.451 | LOCAL | Can try the second floor one. | `campus_services_and_equipment` | `request_information_or_status` |
| 2 | 0.304 | LOCAL | Eh, can help me print also? | `campus_services_and_equipment` | `invite_or_propose` |
| 3 | 0.268 | LOCAL | ok dats fine | `campus_services_and_equipment` | `decline` |

#### MINILM M2

**Selected memory**

- user: There's a printer upstairs.
- peer: Thanks.
- peer: I need to submit a report.
- user: Hope you can print it in time.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.472 | LOCAL | Can try the second floor one. | `campus_services_and_equipment` | `request_information_or_status` |
| 3 | 0.280 | LOCAL | Eh, can help me print also? | `campus_services_and_equipment` | `invite_or_propose` |
| 2 | 0.277 | LOCAL | press yellow button first can already | `campus_services_and_equipment` | `request_information_or_status` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-015

- **Scenario:** `sports_and_exercise`
- **Intent:** `confirm_plan`
- **Incoming message:** peer: still running at 7?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The planned run is at 7.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.667 | LOCAL | 7 lor | `study_sessions_and_spaces` | `request_information_or_status` |
| 3 | 0.423 | LOCAL | Yea, still meeting at the gate. | `meetups_and_transport` | `confirm_plan` |
| 2 | 0.420 | LOCAL | Ya, three still can. | `study_sessions_and_spaces` | `confirm_plan` |

#### BM25 M1

**Selected memory**

- user: That works.
- peer: I have been busy with assignments.
- user: Same here.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.618 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.338 | LOCAL | nah man i got no time, sorry abt that | `assignments_and_group_projects` | `decline` |
| 2 | 0.318 | LOCAL | Can, what time meet? | `study_sessions_and_spaces` | `invite_or_propose` |

#### BM25 M2

**Selected memory**

- peer: Let's go for a run tonight.
- peer: Maybe around 7.
- peer: I have been busy with assignments.
- user: Same here.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.614 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.532 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |
| 3 | 0.485 | LOCAL | idw, i cannot eat spicy | `meals_and_food` | `decline` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.228 | LOCAL | 7 lor | `study_sessions_and_spaces` | `request_information_or_status` |
| 1 | 0.163 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.130 | LOCAL | Yea, five can | `study_sessions_and_spaces` | `confirm_plan` |

#### MINILM M1

**Selected memory**

- user: That works.
- peer: I have been busy with assignments.
- user: Same here.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.198 | LOCAL | ok, then i do my own work | `social_events_and_outings` | `change_or_reschedule` |
| 2 | 0.165 | LOCAL | walao | `meals_and_food` | `change_or_reschedule` |
| 3 | 0.158 | LOCAL | ya, damn long, got like 4 other ppl before us | `campus_services_and_equipment` | `request_information_or_status` |

#### MINILM M2

**Selected memory**

- peer: Let's go for a run tonight.
- peer: Maybe around 7.
- peer: I have been busy with assignments.
- user: Same here.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.258 | LOCAL | Can, eight works for me. | `study_sessions_and_spaces` | `change_or_reschedule` |
| 1 | 0.247 | LOCAL | ya c u later | `sports_and_exercise` | `remind_or_nudge` |
| 3 | 0.244 | LOCAL | of course got, I steady one | `social_events_and_outings` | `remind_or_nudge` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-016

- **Scenario:** `social_events_and_outings`
- **Intent:** `change_or_reschedule`
- **Incoming message:** peer: can we go on Saturday instead?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The museum visit is moved from Sunday to Saturday.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.680 | LOCAL | Yeah, still on for Saturday. | `social_events_and_outings` | `confirm_plan` |
| 2 | 0.457 | LOCAL | ok, another day bah | `assignments_and_group_projects` | `change_or_reschedule` |
| 3 | 0.388 | LOCAL | Can, Friday lor. | `social_events_and_outings` | `change_or_reschedule` |

#### BM25 M1

**Selected memory**

- user: Thanks.
- peer: I also need to buy groceries.
- user: Can do that later.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.617 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.443 | LOCAL | Yeah, still on for Saturday. | `social_events_and_outings` | `confirm_plan` |
| 3 | 0.394 | LOCAL | ok, another day bah | `assignments_and_group_projects` | `change_or_reschedule` |

#### BM25 M2

**Selected memory**

- peer: Let's visit the museum Sunday.
- peer: I'll check the opening hours.
- peer: I also need to buy groceries.
- user: Can do that later.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.628 | LOCAL | Yeah, still on for Saturday. | `social_events_and_outings` | `confirm_plan` |
| 2 | 0.576 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.464 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.417 | LOCAL | uhh saturday sunday can? | `meals_and_food` | `change_or_reschedule` |
| 2 | 0.391 | LOCAL | Yeah, still on for Saturday. | `social_events_and_outings` | `confirm_plan` |
| 3 | 0.259 | LOCAL | ya, tuesday no class | `classes_and_tutorials` | `request_information_or_status` |

#### MINILM M1

**Selected memory**

- user: Thanks.
- peer: I also need to buy groceries.
- user: Can do that later.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.291 | LOCAL | uhh saturday sunday can? | `meals_and_food` | `change_or_reschedule` |
| 1 | 0.290 | LOCAL | ok lor we eat at 2 | `meals_and_food` | `change_or_reschedule` |
| 3 | 0.238 | LOCAL | wat u want | `meals_and_food` | `request_help_or_resource` |

#### MINILM M2

**Selected memory**

- peer: Let's visit the museum Sunday.
- peer: I'll check the opening hours.
- peer: I also need to buy groceries.
- user: Can do that later.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.371 | LOCAL | Yeah, still on for Saturday. | `social_events_and_outings` | `confirm_plan` |
| 2 | 0.333 | LOCAL | uhh saturday sunday can? | `meals_and_food` | `change_or_reschedule` |
| 3 | 0.281 | LOCAL | Can, Friday lor. | `social_events_and_outings` | `change_or_reschedule` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-017

- **Scenario:** `classes_and_tutorials`
- **Intent:** `request_help_or_resource`
- **Incoming message:** peer: can update me on what I missed?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The user offered to share lecture notes or a recap.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.662 | LOCAL | Nevermind, just update me when you reach. | `meetups_and_transport` | `inform_or_update` |
| 2 | 0.579 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |
| 3 | 0.560 | LOCAL | it's okay lah, we update you after rehearsal | `assignments_and_group_projects` | `decline` |

#### BM25 M1

**Selected memory**

- user: Sure.
- peer: I have a group meeting later.
- user: Hope it goes well.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.616 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.419 | LOCAL | Can, I'll go through it with you later. | `study_sessions_and_spaces` | `request_help_or_resource` |
| 2 | 0.408 | LOCAL | idw, i cannot eat spicy | `meals_and_food` | `decline` |

#### BM25 M2

**Selected memory**

- peer: I might miss tomorrow's lecture.
- user: I can share what I write down.
- peer: I have a group meeting later.
- user: Hope it goes well.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.613 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.550 | LOCAL | idw, i cannot eat spicy | `meals_and_food` | `decline` |
| 3 | 0.500 | LOCAL | ya | `study_sessions_and_spaces` | `confirm_plan` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 3 | 0.234 | LOCAL | eh paiseh i got smth on tmr, what about the day after tmr | `study_sessions_and_spaces` | `change_or_reschedule` |
| 2 | 0.193 | LOCAL | ok, then i do my own work | `social_events_and_outings` | `change_or_reschedule` |
| 1 | 0.181 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |

#### MINILM M1

**Selected memory**

- user: Sure.
- peer: I have a group meeting later.
- user: Hope it goes well.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.257 | LOCAL | it's okay lah, we update you after rehearsal | `assignments_and_group_projects` | `decline` |
| 1 | 0.255 | LOCAL | my friends gg too | `social_events_and_outings` | `request_information_or_status` |
| 3 | 0.241 | LOCAL | Abit far leh, but okay la | `study_sessions_and_spaces` | `apologise_or_explain` |

#### MINILM M2

**Selected memory**

- peer: I might miss tomorrow's lecture.
- user: I can share what I write down.
- peer: I have a group meeting later.
- user: Hope it goes well.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.285 | LOCAL | Paiseh, now cannot. I'll send them later. | `course_materials` | `decline` |
| 2 | 0.285 | LOCAL | can, we compare them after class | `course_materials` | `invite_or_propose` |
| 3 | 0.269 | LOCAL | at seminar room. why eh | `classes_and_tutorials` | `request_information_or_status` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-018

- **Scenario:** `meals_and_food`
- **Intent:** `inform_or_update`
- **Incoming message:** peer: I've reached the food court
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The user should acknowledge and coordinate meeting at the food court.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | steady, I meet you all there | `meals_and_food` | `inform_or_update` |
| 3 | 0.578 | LOCAL | ya, reached alr. waiting at the entrance | `sports_and_exercise` | `request_information_or_status` |
| 2 | 0.567 | LOCAL | can lah, small table also okay. i come over now | `meals_and_food` | `inform_or_update` |

#### BM25 M1

**Selected memory**

- user: I'll meet you there.
- peer: The library closes early today.
- user: We should finish our work first.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.471 | LOCAL | Can, what time meet? | `study_sessions_and_spaces` | `invite_or_propose` |
| 3 | 0.426 | LOCAL | steady, I meet you all there | `meals_and_food` | `inform_or_update` |

#### BM25 M2

**Selected memory**

- peer: Let's eat at the food court.
- peer: I'll get there first.
- peer: The library closes early today.
- user: We should finish our work first.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | can lah, small table also okay. i come over now | `meals_and_food` | `inform_or_update` |
| 2 | 0.590 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 3 | 0.575 | LOCAL | idw, i cannot eat spicy | `meals_and_food` | `decline` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.307 | LOCAL | can lah, small table also okay. i come over now | `meals_and_food` | `inform_or_update` |
| 2 | 0.251 | LOCAL | steady, I meet you all there | `meals_and_food` | `inform_or_update` |
| 3 | 0.244 | LOCAL | havent leh, i was waiting for you | `meals_and_food` | `request_information_or_status` |

#### MINILM M1

**Selected memory**

- user: I'll meet you there.
- peer: The library closes early today.
- user: We should finish our work first.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.327 | LOCAL | can, mrt easier. i meet you there | `meetups_and_transport` | `change_or_reschedule` |
| 1 | 0.283 | LOCAL | can lah, small table also okay. i come over now | `meals_and_food` | `inform_or_update` |
| 3 | 0.266 | LOCAL | Ya, three still can. | `study_sessions_and_spaces` | `confirm_plan` |

#### MINILM M2

**Selected memory**

- peer: Let's eat at the food court.
- peer: I'll get there first.
- peer: The library closes early today.
- user: We should finish our work first.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.315 | LOCAL | can lah, small table also okay. i come over now | `meals_and_food` | `inform_or_update` |
| 3 | 0.297 | LOCAL | Paiseh, today cannot. Need finish some work. | `meals_and_food` | `decline` |
| 2 | 0.256 | LOCAL | havent leh, i was waiting for you | `meals_and_food` | `request_information_or_status` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-019

- **Scenario:** `assignments_and_group_projects`
- **Intent:** `request_information_or_status`
- **Incoming message:** peer: did you get a chance to check it?
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: The peer is asking about reviewing their analysis.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.639 | LOCAL | can, i get one for you | `sports_and_exercise` | `request_help_or_resource` |
| 1 | 0.624 | LOCAL | Eh i think need reserve one, wait ah I check | `meals_and_food` | `request_information_or_status` |
| 3 | 0.563 | LOCAL | what you want, i buy for you | `meetups_and_transport` | `request_information_or_status` |

#### BM25 M1

**Selected memory**

- user: I need to check the calculations.
- peer: Our presentation is next week.
- user: We still have time to polish it.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.628 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.490 | LOCAL | sorry ah but i still need it leh, I can only return by next week | `campus_services_and_equipment` | `change_or_reschedule` |
| 3 | 0.437 | LOCAL | Eh i think need reserve one, wait ah I check | `meals_and_food` | `request_information_or_status` |

#### BM25 M2

**Selected memory**

- peer: Can you review my analysis?
- user: I need to check the calculations.
- peer: Our presentation is next week.
- user: We still have time to polish it.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.624 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.478 | LOCAL | sorry ah but i still need it leh, I can only return by next week | `campus_services_and_equipment` | `change_or_reschedule` |
| 3 | 0.402 | LOCAL | Eh i think need reserve one, wait ah I check | `meals_and_food` | `request_information_or_status` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.172 | LOCAL | I steady one, I checked the information. | `course_materials` | `confirm_plan` |
| 2 | 0.171 | LOCAL | ya confirm legit | `classes_and_tutorials` | `confirm_plan` |
| 3 | 0.169 | LOCAL | eh paiseh i got smth on tmr, what about the day after tmr | `study_sessions_and_spaces` | `change_or_reschedule` |

#### MINILM M1

**Selected memory**

- user: I need to check the calculations.
- peer: Our presentation is next week.
- user: We still have time to polish it.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 2 | 0.236 | LOCAL | got, i send you in a bit | `course_materials` | `request_information_or_status` |
| 3 | 0.234 | LOCAL | uhh, i think still need do slides | `course_materials` | `remind_or_nudge` |
| 1 | 0.209 | LOCAL | oh ok nvm all g | `assignments_and_group_projects` | `decline` |

#### MINILM M2

**Selected memory**

- peer: Can you review my analysis?
- user: I need to check the calculations.
- peer: Our presentation is next week.
- user: We still have time to polish it.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.244 | LOCAL | I steady one, I checked the information. | `course_materials` | `confirm_plan` |
| 3 | 0.231 | LOCAL | Can, I check later. | `assignments_and_group_projects` | `request_help_or_resource` |
| 2 | 0.206 | LOCAL | oh ok nvm all g | `assignments_and_group_projects` | `decline` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

### DIAG-LC-020

- **Scenario:** `meetups_and_transport`
- **Intent:** `apologise_or_explain`
- **Incoming message:** peer: sorry, took the wrong bus
- **Expected context (draft note):** LONG-CONTEXT DIAGNOSTIC; draft case requiring team review. Expected relevant context: Acknowledge the delay and ask them to update when they arrive.

#### BM25 M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | Neh mind, happens one | `meals_and_food` | `apologise_or_explain` |
| 2 | 0.504 | LOCAL | ya, same one as yesterday | `meetups_and_transport` | `confirm_plan` |
| 3 | 0.422 | LOCAL | Nevermind, just update me when you reach. | `meetups_and_transport` | `inform_or_update` |

#### BM25 M1

**Selected memory**

- user: The bus is delayed.
- peer: I can wait a bit.
- user: Thanks for understanding.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.560 | LOCAL | No, I taking bus, MRT very slow | `meetups_and_transport` | `request_information_or_status` |
| 3 | 0.432 | LOCAL | ya, same one as yesterday | `meetups_and_transport` | `confirm_plan` |

#### BM25 M2

**Selected memory**

- peer: I'm waiting at the campus gate.
- user: The bus is delayed.
- peer: I can wait a bit.
- user: Thanks for understanding.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.600 | LOCAL | yah have to , they change so much | `assignments_and_group_projects` | `confirm_plan` |
| 2 | 0.562 | LOCAL | Yea, still meeting at the gate. | `meetups_and_transport` | `confirm_plan` |
| 3 | 0.513 | LOCAL | No, I taking bus, MRT very slow | `meetups_and_transport` | `request_information_or_status` |

#### MINILM M0

**Selected memory**

- None (M0 / no selected memory)

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.259 | LOCAL | ya, same one as yesterday | `meetups_and_transport` | `confirm_plan` |
| 2 | 0.202 | LOCAL | Nevermind, just update me when you reach. | `meetups_and_transport` | `inform_or_update` |
| 3 | 0.199 | LOCAL | walao | `meals_and_food` | `change_or_reschedule` |

#### MINILM M1

**Selected memory**

- user: The bus is delayed.
- peer: I can wait a bit.
- user: Thanks for understanding.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 1 | 0.243 | LOCAL | Nevermind, just update me when you reach. | `meetups_and_transport` | `inform_or_update` |
| 2 | 0.232 | LOCAL | kk , take ur time ah, i walk ard first. | `meetups_and_transport` | `apologise_or_explain` |
| 3 | 0.211 | LOCAL | ya, same one as yesterday | `meetups_and_transport` | `confirm_plan` |

#### MINILM M2

**Selected memory**

- peer: I'm waiting at the campus gate.
- user: The bus is delayed.
- peer: I can wait a bit.
- user: Thanks for understanding.

**Top candidates**

| Rank | Score | Style | Candidate reply | Scenario | Intent |
|---:|---:|---|---|---|---|
| 3 | 0.264 | LOCAL | sorry ah i reaching in 5 mins, got train delay | `assignments_and_group_projects` | `request_information_or_status` |
| 2 | 0.263 | LOCAL | kk , take ur time ah, i walk ard first. | `meetups_and_transport` | `apologise_or_explain` |
| 1 | 0.257 | LOCAL | Nevermind, just update me when you reach. | `meetups_and_transport` | `inform_or_update` |

**Human review**

- **Best contextual fit among the six configurations:** _to be reviewed_
- **Does retrieved memory preserve the relevant plan/detail?** _to be reviewed_
- **Notes / failure mode:** _to be reviewed_

---

