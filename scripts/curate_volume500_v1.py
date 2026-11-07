"""Apply the recorded assistant audit to this frozen synthetic batch only.

This is a one-off data revision, not an automatic semantic validator. Original
model outputs and reviews remain in artifacts/volume500_v1/full unchanged.
"""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/volume500_v1/full/reviewed.jsonl'
DEST = ROOT / 'data/synthetic/volume500_v1_reviewed.jsonl'

# Index: STANDARD, LOCAL. These are explicit editorial decisions following
# inspection of every history, incoming peer message, and paired reply.
REPLIES = {
0: ("I'll meet you at the lobby.", "I'll meet you at lobby then."),
1: ("We can order together another day.", "Next time then order together."),
2: ("Let's meet at 2 PM at the main entrance.", "Meet at 2pm at the main entrance then."),
6: ("Thanks, I'll check them out and let you know.", "Got it, let me check the file then update you."),
7: ("I'll check the booking and get back to you.", "Let me check if room still free, then update you."),
25: ("Yes, I'll be there around 9am.", "Coming, will reach around 9am."),
28: ("Thanks, let me know once you've added it.", "Thanks, tell me once you add it in."),
30: ("Thanks for handling the chapter summary too.", "Thanks for doing the chapter summary also."),
31: ("Got it, I'll head to the lecture hall.", "Noted, I'll go lecture hall then."),
33: ("The notes from Tuesday cover the main theories we need.", "Tuesday's notes got the main theories we need."),
34: ("I think they close at 6 PM, so you should make it.", "I think close at 6pm, you should still make it."),
39: ("I'll return it before three as planned.", "I'll return before three like we said."),
41: ("I'll check the content and let you know.", "Let me check the content first, then update you."),
42: ("I'll check the draft and let you know about the font size.", "Let me check the draft, then tell you if font size okay."),
46: ("I haven't received them yet, I'll check my folder now.", "Still haven't got them, let me check my folder first."),
48: ("I can do that for you since I'm nearby.", "Can, I'm nearby anyway."),
51: ("Oh, I see. I'll head over to the locker to get it.", "Oh okay, I go locker collect it now."),
52: ("They usually close at 6pm, so you should be fine.", "Usually close at 6pm, so you should still have time."),
63: ("I don't think we have enough stock, but we could try a small batch.", "I think stock not enough, but maybe can try small batch."),
64: ("Good idea, the bus route is usually more consistent than the train.", "Good idea, bus route usually more steady than train."),
67: ("Yes, bring it up and I'll keep it for you.", "Can, bring up and I'll keep for you first."),
68: ("No problem, I'll update the order to 30% sugar.", "Can, I'll change yours to 30% sugar."),
73: ("Horror works for me. How about the new one?", "Horror can. Want to watch the new one?"),
78: ("I haven't finished them yet, but I'll send them over later tonight.", "Not done yet, let me finish then send you tonight."),
81: ("Next Tuesday works. Let's go then.", "Next Tuesday can. Let's go then."),
83: ("Thanks, please send the photos of us by the stage.", "Thanks, send me the ones of us by stage."),
93: ("No worries about the lift. Coffee later sounds good.", "No worries about the ride. Coffee later can."),
99: ("No worries, good luck with your quiz tomorrow.", "No problem, all the best for your quiz tomorrow."),
108: ("I'll handle the ticket booking. Let me know the time.", "I'll book the tickets, you tell me the time."),
137: ("I'm going to the indie band one, which starts at eight.", "Going for indie band one, start at eight."),
140: ("I'll send just the summary since the slides are huge.", "I'll send summary only, slides too big."),
141: ("Thanks for spotting it. I'll check the lobby now.", "Ah thanks, I go lobby now to check."),
144: ("The lounge is too noisy for me. Let's try the level three study pods.", "Lounge too loud for me. Better try level three study pods."),
151: ("Yes, everyone else has confirmed nine works for them too.", "Yes, everyone else confirm nine can also."),
155: ("No worries, stay with your current group then.", "No problem, you stay with your group then."),
156: ("That works, I only need help with the layout.", "Can, I only need help with layout."),
157: ("Yes, there's enough room for large containers.", "Got space, big containers can fit."),
158: ("Four works for me. Thanks for updating the calendar.", "Four can. Thanks for updating the calendar."),
160: ("I'll get it at five thirty and let you know.", "I'll collect at five thirty, then update you."),
169: ("Yes, I'll be there at the registration booth.", "Confirm, I'll be there at registration booth."),
183: ("I'd prefer the gym lockers. Could you leave it there?", "Better leave at gym lockers. Can put there instead?"),
184: ("Great, please collect it when you can.", "Good, go collect when you're free."),
186: ("Bring it along; I can bring mine too.", "You bring first, I can bring mine also."),
187: ("Six works. I'll make sure we're done by seven thirty.", "Six can. I'll make sure we finish by seven thirty."),
188: ("Logistics would be helpful. I'll find someone for registration.", "Logistics can help. I'll find someone for registration."),
189: ("Yes, let's print my draft there instead.", "Can, let's print my draft there instead."),
192: ("The stall at Block Four is the best for chicken rice.", "Block Four stall best for chicken rice."),
193: ("Okay, please return it by four.", "Can, return to me by four then."),
197: ("Got it, I'll see you at the library.", "Noted, see you at library then."),
199: ("Thanks, I'll check near the bin for your bottle.", "Thanks, I go check near bin for your bottle."),
205: ("I'll put the logo on the top right.", "I'll put logo at top right then."),
206: ("No worries, I'll find someone else for the crates.", "No problem, I'll find another person for crates."),
207: ("Tomorrow evening works for me.", "Tomorrow night can for me."),
210: ("I'll be the one driving there to collect the equipment.", "I'll drive there to collect the equipment."),
212: ("No worries. Could you help with Friday's preparation instead?", "No problem. Can help with prep on Friday instead?"),
213: ("Could you upload the summit shots too? I can't find them either.", "Can upload the summit ones also? I can't find them either."),
223: ("Let's meet at the student lounge instead; it should be quieter.", "Better meet student lounge instead, should be more quiet."),
234: ("I can highlight those sections once you send the notes.", "Can highlight those parts once you send me the notes."),
237: ("Yes, it's still on for tomorrow morning.", "Still on tomorrow morning, see you there."),
240: ("Coffee on Friday works for me.", "Friday go coffee can for me."),
241: ("Got it, I'll arrange ten chairs for the setup.", "Noted, I'll get ten chairs for setup."),
242: ("Five at the void deck works for me.", "Five at void deck can for me."),
244: ("Let's start the setup first and wait for him to join.", "Start setting up first, wait for him to join later."),
249: ("I'm fine with my umbrella, thanks. Let's head off.", "My umbrella can already, thanks. Let's go."),
250: ("I'll check whether another court at the sports hub is available.", "Let me check if sports hub got another court free."),
251: ("I think it finishes around seven thirty.", "I think finish around seven thirty."),
252: ("I'm happy to take on logistics and transport too.", "Can, I'll handle logistics and transport also."),
253: ("Five extra packs should be enough for everyone.", "Five extra packs should be enough for all."),
269: ("Yes, I have the cables and the sensor ready.", "Got the cables and sensor ready already."),
275: ("I'm at Exit B, right near the coffee shop.", "I'm at Exit B, near the coffee shop."),
279: ("I understand. Let me know if you can share them later.", "No worries, tell me if you can send later."),
280: ("Sure, you can keep it until the end of the day.", "Can, keep until end of today then."),
281: ("Just water, please. I'll pick it up with the groceries.", "Water only, thanks. I'll get with the groceries."),
285: ("No worries, I'll handle the conclusion slides.", "No problem, I'll do the conclusion slides then."),
286: ("I might have. I'll check the printer area next.", "Maybe got. I'll check printer area next."),
297: ("Chicken rice sounds good. Let's try that stall.", "Chicken rice can. Let's try that stall."),
299: ("I have the elasticity notes. I'll send that part over.", "Got the elasticity notes, I'll send that part to you."),
300: ("That works. I'll manage the merchandise booth.", "Can, I'll handle merchandise booth then."),
306: ("I can print five copies for you this afternoon.", "Can, I'll print five copies for you this afternoon."),
308: ("I'd keep the headings as they are and change the body text to Arial.", "Headings keep same, body text change to Arial."),
309: ("Proofreading is fine; please check the grammar and flow.", "Proofreading can, help check grammar and flow."),
310: ("Agreed, let's use last night's slide.", "Agree, use last night's slide then."),
315: ("Then keep the tripod for now; it doesn't need returning today.", "Then keep tripod first, no need return it today."),
318: ("Yes, please handle the bibliography and formatting.", "Can, you do bibliography and formatting."),
325: ("I'm planning to go, see you there!", "Going, see you there later!"),
327: ("I'm not sure they'll accept late entries. I'll check for a waiting list.", "Not sure if they accept late entries. I'll check waiting list first."),
332: ("Joining your study group sounds good. When are you meeting?", "Join your study group can. You all meeting when?"),
340: ("Probably. I'll check the online catalog for availability.", "Likely. Let me check online system if it's available."),
341: ("Thanks, let me know once you've checked with the group.", "Thanks, update me once you check with group."),
342: ("The shop closes at four thirty. Could someone else collect them earlier?", "Shop closes four thirty. Can someone else collect earlier?"),
351: ("Send what you have; I'll find the missing part elsewhere.", "Send what you got first, I'll find the missing part elsewhere."),
357: ("Eight works, thanks for offering me a lift.", "Eight can, thanks for giving me a ride."),
365: ("I don't have an umbrella. Shall we wait for the rain to stop?", "Don't have umbrella. Wait for rain to stop first?"),
374: ("Four works if we're collecting it tomorrow.", "Four can if we're collecting tomorrow."),
379: ("Got it, I'll arrange the four chairs and two tables.", "Noted, I'll get four chairs and two tables."),
380: ("I can go down and bring it up myself.", "I go down bring up myself then."),
382: ("I can do three, but I'll miss the first part.", "Three can, but I'll miss the first part."),
388: ("Let's meet at the library on Tuesday evening.", "Tuesday evening meet at library then."),
393: ("Sunday works. Let's go then.", "Sunday can. Let's go then."),
395: ("Could you add some oranges and milk? I'll get the rest.", "Can get some oranges and milk? I'll buy the rest."),
400: ("Yes, I still have it. I'll return it tomorrow.", "Still got it, will return to you tomorrow."),
402: ("If I make it, I'll help take some photos for you.", "If I go, I'll help you take photos."),
405: ("I can't, as I'm already on my way to the library.", "Cannot, I'm already on the way to library."),
409: ("I'll print them single-sided, then.", "Then I just print one side only."),
427: ("Smart casual is fine; there's no formal dress code.", "Smart casual can, no formal dress code."),
429: ("That's too late for me. Could someone else collect it before six?", "Too late for me. Can someone else collect before six?"),
430: ("I'll leave it at your door when I get there.", "I'll put at your door when I reach."),
441: ("We have enough chairs, but could you bring snacks instead?", "Chairs enough already, can bring snacks instead?"),
442: ("True, but it's the only place with enough outlets.", "Yeah, only there got enough power points."),
445: ("I checked and it's free until five.", "Checked already, free till five."),
453: ("I'm good for now, but maybe we can buy bread next week?", "I'm okay for now, maybe next week buy bread together?"),
459: ("You're right, I'll add clay to the supply list.", "Right, I'll add clay to the supply list now."),
465: ("I really can't make it today, sorry about that.", "Today really cannot make it, sorry."),
467: ("I'm still waiting for the design team to send it.", "Still waiting for design team to send final file."),
479: ("I'll check my room and keep it aside for you.", "I'll check my room, keep for you first."),
487: ("That makes sense, trains are more reliable during peak hours.", "True, train more reliable during peak hours."),
489: ("I can't go there now, but do you want to try the cafe?", "Cannot go there now, want to try cafe instead?"),
493: ("No, it opens this Friday at ten on the student portal.", "Not yet, open this Friday ten on student portal."),
494: ("I can't finish by five. Could I return it at six instead?", "Cannot finish by five. Can return at six instead?"),
498: ("Could you help with stage decorations? I'll handle the sound check.", "Can help with stage deco? I'll do sound check."),
}
MESSAGES = {
183: "Can I just leave it at the common area instead?",
193: "Can I keep it until four? I need it before I head home.",
253: "I got five extra packs. Is that enough?",
286: "Still no luck? Did you leave it near the printer earlier?",
357: "Is eight still okay for me to pick you up?",
430: "Thanks. Can you leave it at my door?",
440: "Can you still walk with me, or do you need to catch that bus?",
}


def main():
    original = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    assert len(original) == 500
    revised = copy.deepcopy(original)
    audit = []
    for i, c in enumerate(revised):
        changes = []
        # Generation frequently copied the incoming message into the history.
        while c['session_memory'] and c['session_memory'][-1]['speaker'] == 'peer' and c['session_memory'][-1]['text'].strip().casefold() == c['message'].strip().casefold():
            c['session_memory'].pop()
            changes.append('Removed duplicate final peer history turn.')
        if i == 350:
            c['session_memory'][1]['text'] = "Two is too early for me. Can you make three instead?"
            changes.append('Fixed internally contradictory earlier/later wording in fictional history.')
        if i in MESSAGES:
            c['message'] = MESSAGES[i]
            changes.append('Revised fictional incoming message to preserve ownership/actor continuity.')
        if i in REPLIES:
            c['replies'] = dict(zip(('standard', 'local'), REPLIES[i]))
            changes.append('Edited paired targets for role continuity, an actual response, or matching commitments/details/certainty; see before/after.')
        prior_review = copy.deepcopy(c['review'])
        c['review'] = dict(reviewer='Codex assistant full-batch editorial audit; not human or an independent blinded evaluation',
            meaning_preserved=True, speaker_correct=True, natural_local=True, facts_supported=True,
            scope='Fictional selectable drafts; plausible new details allowed; no world-truth attestation.',
            prompt_version='volume500-assistant-audit-v1', original_model_review=prior_review)
        c['provenance']['editorial_revision'] = 'volume500-assistant-audit-v1'
        audit.append(dict(index=i, id=c['id'], decision='revise' if changes else 'keep', reasons=changes,
                          before={k: original[i][k] for k in ('session_memory','message','replies')},
                          after={k: c[k] for k in ('session_memory','message','replies')}))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in revised))
    out = SOURCE.parent / 'assistant_audit.json'
    report = dict(status='complete', inspected=500, reviewer='Codex assistant; not human; not blind to generation process',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        revised_sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
        target_edits=len(REPLIES), incoming_edits=len(MESSAGES), history_logic_edits=1,
        items_revised=sum(a['decision']=='revise' for a in audit),
        limitations='Assistant editorial judgment is fallible. Same-model automatic approval was insufficient. This audit is training-data curation, not an evaluation score.',
        items=audit)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='items'},indent=2))


if __name__ == '__main__':
    main()
