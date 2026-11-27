"""Write an explicit, reviewable target overlay; no source-record mutation."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/sealion_rewrite_targets_v1'

# Indices refer to the frozen, prioritised candidate_pool.json, not model outputs.
# Short/already-natural examples are intentionally kept unchanged.
UNCHANGED={16,20,28,30,35,39,40,41,45,49,53,61,65,66,69,81,84,88,109,117}
REVISIONS={
0:'Updated song list send me before practice, please.',
1:'Five copies can, this afternoon I print for you.',
2:'At the north gate, just beside the blue vending machine.',
3:'I can help you ask around for the elasticity notes.',
4:'I add in the competitor analysis page, then send the updated slides.',
5:'No worries, I can update you on whatever you miss.',
6:'Thanks! I go check near the bin for your bottle.',
7:'Thanks! I go check Telegram now.',
8:'I put in the budget slides, then send you the updated file.',
9:'Yes please, help me handle logistics and equipment setup.',
10:'Ya, I think stay home rest better.',
11:'I come pick you up at the station exit.',
12:'Thanks for checking! We start a group chat with Sarah?',
13:'White background can use. The file I update now.',
14:'Logistics would help. Registration side I find someone else.',
15:'Yes please, put those graphs into the final layout.',
17:'His part of the experiment I can take over for you.',
18:'Yes please, installation you guide me through can?',
19:'Check your guard house first? Parcel there then let me know.',
21:'I haven’t check the bin yet. I go now.',
22:'I send finance the receipt photo.',
23:'Dunno how much space left. Before bringing the containers, check first.',
24:'Your water bottles I definitely can help carry.',
25:'I go get us a table near the window.',
26:'Okay, your seat I won’t reserve then.',
27:'I put them in the equipment cage already.',
29:'Yes please, help bring the posters for the booth.',
31:'Yes please, the form email me one copy.',
32:'Thanks! I go see the folder now.',
33:'Whatever you have send first. Missing part I find somewhere else.',
34:'I add the dumplings into your laksa order.',
36:'Dunno if they still accept late entries. I check got waiting list anot.',
37:'Today I taking bus, but I can walk with you if you want.',
38:'Stage deco you help? Sound check I settle.',
42:'Dunno what time the studio closes. Before you go, check first.',
43:'I remove the blur photos, then update the folder tonight.',
44:'The timing I not sure. Let me check the schedule first.',
46:'Paiseh, I promised to keep it within our group.',
47:'Dunno what was said about the midterm notes. Ask the prof first?',
48:'Booking details I check first, then update you about the projector.',
50:'Exit number or location pin send me can? Then I can find you.',
51:'Then I go check the lost and found office.',
52:'Ya, you’re right. I add clay to the supply list.',
54:'No, after our shift I returned it to the event desk already.',
55:'I haven’t check the schedule yet. We see the portal or notice board first?',
56:'Find another study area with more comfortable chairs can?',
57:'Ride no worries, later coffee sounds good.',
58:'Thanks for handling the chapter summary also!',
59:'I didn’t see it. But lost and found I go check.',
60:'Good idea, I go check the bus timings for my route.',
62:'We start setting up first, wait for him to join.',
63:'Final headcount I not sure. I check with the group.',
64:'I share the Monday notes with you, last part I help summarise also.',
67:'Not sure, that section I haven’t check yet.',
68:'Got it, I arrange ten chairs for the setup.',
70:'I cannot drive. Someone else drive us can?',
71:'I check who driving to collect the equipment, then update you.',
72:'Got it, I go straight to the gym for practice.',
73:'Proofreading can. Grammar and flow help me check.',
74:'Sure, the schedule I forward to you.',
75:'Aiyo, sorry to hear that. Want to talk about it?',
76:'Thanks! How to book one, you explain can?',
77:'Popcorn and snacks you get? I book the tickets.',
78:'Tripod you keep first, today no need return.',
79:'See you at the library. I bring the notes like we planned.',
80:'I add some oranges and milk to the shopping list.',
82:'Data analysis section you take? Intro and conclusion I do.',
83:'Yes please, your extra ruler lend me can?',
85:'Then the pasta place I suggested leh?',
86:'I put the logo at the top right.',
87:'Yes please, tonight bring my raincoat.',
89:'Dress code I not sure. We check the event announcement first?',
90:'We find a translated version of the instructions?',
91:'Find a laksa stall instead?',
92:'Meet at three can. But the tutorial itself still at two?',
93:'Sure, how to use it I show you.',
94:'Three works. Shift change we confirm with the organiser first?',
95:'I bringing my ID down. Tell the staff I collect it myself can?',
96:'My goggles you return at the pool can.',
97:'Dunno which other lift available. We check the signs first?',
98:'No worries, thanks for checking my bag!',
99:'Still deciding. Waiting times I compare first.',
100:'The last hour of practice, come join us!',
101:'I check my bag first. If I have it, I bring to the library.',
102:'Like that we miss the last shuttle. Leave before ten can?',
103:'My parcel I can go down and collect.',
104:'Dunno registration open anot. Check the club announcement first?',
105:'No problem, I leave it at the front desk instead.',
106:'I check we’re done with the projector first, then arrange return before five.',
107:'Sure, I send you the notes without the last section.',
108:'Thanks for checking! I go download it now.',
110:'I check the tracking number, then send it to you.',
111:'I checking the portal now. Registration closed anot, I confirm first.',
112:'Ya, still with me. Tomorrow I return it.',
113:'Before borrowing the books, check how many people need a copy first.',
114:'I add in the QR code and change the theme to navy blue.',
115:'No worries, all the best for tomorrow’s quiz!',
116:'Three I can meet, but orientation starts at two. Check can join late anot?',
118:'Ya, entrance there easier for people to notice. We book that spot?',
119:'Then we just focus on the practice problems in that slide deck.',
}


def main():
    source=json.loads((OUT/'candidate_pool.json').read_text())
    assert len(source)==120 and len(REVISIONS)==100 and len(UNCHANGED)==20
    assert set(REVISIONS).isdisjoint(UNCHANGED) and set(REVISIONS)|UNCHANGED==set(range(120))
    overlay=[]
    for i,row in enumerate(source):
        assert row['item_id'].startswith('SYN-')
        revised=REVISIONS.get(i,row['response'])
        assert revised.strip() and ((revised!=row['response'])==(i not in UNCHANGED)),i
        overlay.append(dict(row,new_response=revised,changed=i not in UNCHANGED,
            annotation='Assistant-authored style alternative; independently human-reviewed naturalness not established.',
            review_criterion='Keep the English speaker, people, facts, time, modality and decision. Moderate chatty local phrasing; no mandatory particles or Malay substitutions.',
            edit_note='Targeted wording alternative; old wording is not automatically judged wrong.' if i not in UNCHANGED else 'Already-natural target retained as a no-change control.'))
    text=json.dumps(overlay,indent=2,ensure_ascii=False)+'\n'
    path=OUT/'overlay.json'
    if path.exists() and path.read_text()!=text:raise FileExistsError(path)
    path.write_text(text)
    preview=[overlay[i] for i in [0,1,2,4,5,10,16,23,27,38,50,62,70,83,117]]
    (OUT/'style_preview.json').write_text(json.dumps(preview,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(overlay=len(overlay),revised=sum(r['changed'] for r in overlay),unchanged_controls=20)))


if __name__=='__main__':main()
