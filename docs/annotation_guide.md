# Annotation guide

Audience: the five of you. This is the Role A Week 4 artefact.

## Task

Write a short incoming message from a peer, plus a little recent chat, then author two replies:

- **STANDARD**: clear international English an exchange student can send safely.
- **LOCAL**: English-led, lightly local. May echo a Mandarin or Singlish term that already appeared. Never invent code-switching. Never pile on slang.

Scope: casual peer coordination only (makan, meeting, running late, library, notes, rain, supper). Not customer service, not dating, not academic tutoring, not admin/official email.

## What the labels mean

- **Scenario** is the dominant real-world topic governing the incoming message and its reply.
- **Incoming intent** is what the latest peer message is trying to do. It never describes the reply.
- **Recent conversation** is prior context only. An incidental topic in an older turn does not determine the scenario.

When several topics appear, choose the one directly relevant to the incoming message and reply. Weather, weekends, lateness, and group chat are context rather than scenarios.

### Scenarios

| Scenario | Use it when the exchange is mainly about |
| --- | --- |
| `meals_and_food` | Choosing, obtaining, or meeting for food or drinks |
| `study_sessions_and_spaces` | Arranging peer study or finding and sharing study space |
| `course_materials` | Requesting, locating, or sharing notes, slides, readings, or recordings |
| `assignments_and_group_projects` | Coordinating assessed work, contributions, deadlines, or presentations |
| `classes_and_tutorials` | Coordinating attendance, time, room, or changes to scheduled teaching |
| `meetups_and_transport` | Travelling, arriving, waiting, or locating one another for a meetup |
| `social_events_and_outings` | Coordinating non-food, non-sport leisure activities, clubs, or outings |
| `sports_and_exercise` | Coordinating participation in sport, exercise, or physical recreation |
| `campus_services_and_equipment` | Using printers, queues, chargers, devices, or shared equipment |

Boundary rules:

- Asking for a notes file is `course_materials`; dealing with a broken printer is `campus_services_and_equipment`.
- Finding a library seat is `study_sessions_and_spaces`; asking about assignment progress is `assignments_and_group_projects`.
- Asking for a tutorial room is `classes_and_tutorials`; asking where a friend is during a meetup is `meetups_and_transport`.

### Incoming intents

| Intent | The incoming peer message |
| --- | --- |
| `invite_or_propose` | Suggests a new activity, plan, or option |
| `request_information_or_status` | Asks for a fact, location, progress, readiness, or current state |
| `request_help_or_resource` | Asks the recipient to perform a new action or provide something |
| `confirm_plan` | Confirms or checks an arrangement already discussed |
| `change_or_reschedule` | Asks to modify an existing arrangement |
| `decline` | Refuses or says they cannot participate in a request, invitation, or plan |
| `inform_or_update` | Provides new information without primarily requesting action or apologising |
| `remind_or_nudge` | Prompts completion of an obligation already established in the conversation |
| `apologise_or_explain` | Apologises for or explains a problem, absence, failure, or delay |

For commonly confused requests, ask what the peer primarily wants:

- An answer or current state → `request_information_or_status`
- A new action or resource → `request_help_or_resource`
- Completion of an existing obligation → `remind_or_nudge`; the earlier conversation must establish that obligation

## Item recipe

1. Pick a canonical scenario and incoming intent using the definitions above.
2. Write 1–3 prior turns. Newest last. No real names, phones, student IDs, hostel room numbers, or course codes that identify a person.
3. Write the incoming `message`. It may mix English, Singlish particles, and a small Mandarin word/phrase in characters or pinyin.
4. Write `replies.standard` first. If you cannot say it in plain English, the item is out of scope.
5. Write `replies.local` from the standard reply: shorten, keep English word order, optionally keep one existing local term.
6. Tag `language_mix` and `local_terms`.
7. Set `privacy_ok: true` only after a self-check.

## Style rules

Allowed in LOCAL: lah / leh / lor / can / paiseh / makan / jio, and echoing 吃饭 / 到了 / 谢谢 when the message already used them.

Not allowed: forcing three particles into one line; adding Mandarin that the peer never used; caricature ("blur like sotong, die die must makan walao sia"); translating the whole sentence into Chinese.

If the incoming message is already standard English, LOCAL stays almost the same. Do not "Singlish-ify" it for fun.

## Quality bar

Good STANDARD: "Want to grab lunch first and then head to the library?"  
Good LOCAL: "Can, we makan first then library lah."  
Bad LOCAL: "Wah shiok we 吃饭 then 图书馆 kiasu catch seats walao."

## Review checklist

Reviewer rejects the item if any box fails:

- [ ] Privacy-safe
- [ ] Casual peer coordination
- [ ] STANDARD is natural English
- [ ] LOCAL is English-led and restrained
- [ ] LOCAL does not invent Mandarin
- [ ] Two replies are not identical unless the input is already standard English
- [ ] Memory is actually useful for the reply
- [ ] No LLM-generated wording pasted in

Record the independent reviewer's initials in `reviewer`. The legacy `second_reviewer` field remains compatible but is not required for this cycle. Saving a valid review automatically sets the item's `status` to `completed`. Split labels stay `dev` or `test`; Role A assigns splits, authors do not self-assign test items after seeing model output.
