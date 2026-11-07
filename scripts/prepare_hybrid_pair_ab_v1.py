"""Fresh conversational inputs for paired-model comparison; no reference replies."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/human_review/hybrid_pair_ab_v1'


def fresh_cases():
    scenarios=[
('food',[('peer','Done with your tutorial?'),('user','Yes, finally finished.')],'Want to get dinner at the food court beside the station?'),
('food',[('peer','I am heading to the cafe downstairs.'),('user','I might get a drink too.')],'Do you feel like coffee or something cold?'),
('food',[('peer','Have you eaten anything today?'),('user','Only a sandwich this morning.')],'What kind of food are you in the mood for tonight?'),
('food',[('peer','We can grab supper after the film.'),('user','Sounds good, I know a place near the cinema.')],'Which supper place did you have in mind?'),
('food',[('peer','The canteen is packed right now.'),('user','We could try somewhere less crowded.')],'How about the noodle shop across the road instead?'),
('care',[('peer','I booked a clinic appointment for this afternoon.'),('user','What are you getting checked?')],'My eyelid has been bothering me, so I want the doctor to look at it.'),
('care',[('peer','I have an interview first thing tomorrow.'),('user','How are you feeling about it?')],'Quite nervous honestly, I keep worrying that I will blank out.'),
('care',[('peer','The presentation did not go as planned.'),('user','Oh no, what happened during it?')],'I forgot a section halfway through and felt so embarrassed.'),
('care',[('peer','I will be late because I missed my bus.'),('user','No problem, I am still at the cafe.')],'Thanks for waiting, today has been a bit of a mess.'),
('care',[('peer','I got the internship offer!'),('user','You were waiting for that email all week.')],'I am so relieved that they finally got back to me.'),
('coordination',[('peer','I have arrived at the sports hall.'),('user','I am walking over from the bus stop.')],'Should I wait by the ticket counter or go inside first?'),
('coordination',[('peer','Where should we meet before the concert?'),('user','I will wait at the entrance facing the river.')],'Which entrance are you waiting at again?'),
('coordination',[('peer','Is everyone ready for the group photo?'),('user','I am here, but Maya has not arrived yet.')],'Do you want to wait a few minutes for Maya?'),
('coordination',[('peer','I could not find the room for our discussion.'),('user','It is the small room beside the lift.')],'Ah found it, are you already inside?'),
('coordination',[('user','I will get us a table near the window.'),('user','Message me when you enter the cafe.')],'I just walked in through the side door, where are you sitting?'),
('study',[('peer','I wrote a draft of the introduction.'),('user','I can help you look through it.')],'Could you check whether the second paragraph is too repetitive?'),
('study',[('peer','Our slides need to be uploaded before midnight.'),('user','I have already uploaded my section.')],'Please remember to upload your slides tonight.'),
('study',[('peer','Have you managed to start the practice questions?'),('user','I finished the first two but got stuck on the third.')],'Which question are you stuck on exactly?'),
('study',[('peer','The submission deadline has moved to next Monday.'),('user','That gives us more time to check the references.')],'Should we use tomorrow to go through the references together?'),
('study',[('peer','I missed the first half of the lecture.'),('user','I took notes from the beginning.')],'Could you tell me what the lecturer covered before I arrived?'),
('constraints',[('peer','What time can you reach the station tonight?'),('user','I cannot get there before half past seven.')],'Would meeting at seven sharp be okay for you?'),
('constraints',[('peer','Can we plan our project meeting this week?'),('user','I am busy all Tuesday, but Wednesday afternoon is free.')],'Shall we meet on Tuesday after lunch then?'),
('constraints',[('peer','Let us choose a place for dinner.'),('user','I need to keep dinner under fifteen dollars today.')],'Are you okay with the thirty-dollar set menu?'),
('constraints',[('peer','How long can you stay at the exhibition?'),('user','I need to leave by three to pick up my sister.')],'Could you stay until four and help pack up?'),
('constraints',[('peer','Would you like to join us for the weekend trip?'),('user','I cannot make it this weekend, but I am free the weekend after.')],'Can you come along this Saturday morning?'),
('ownership',[('peer','Thanks for lending me your portable fan.'),('user','No problem, you can use it during the event.')],'Can I return your fan tomorrow instead of this evening?'),
('ownership',[('peer','Did my spare umbrella help yesterday?'),('user','Yes, I still have your umbrella with me.')],'Could you bring my umbrella back when we meet tonight?'),
('ownership',[('peer','I think I left my notebook on your desk.'),('user','Yes, your notebook is beside my laptop.')],'Do you still have my notebook there?'),
('ownership',[('peer','I borrowed your HDMI adapter for class.'),('user','I will need my adapter for a presentation tomorrow.')],'Should I return your adapter before your presentation?'),
('ownership',[('peer','You can borrow my tote bag for the books.'),('user','Thanks, I will bring your tote bag home today.')],'Could you return my tote bag at our next study session?'),
]
    return [dict(item_id=f'PAIR-FRESH-{n:03}',category=category,context=[t for _,t in turns],speakers=[s for s,_ in turns],message=message,response='') for n,(category,turns,message) in enumerate(scenarios,1)]


def normalize(text):return re.sub(r'\W+',' ',text.casefold()).strip()


def prepare(folder=FOLDER):
    if folder.exists():raise FileExistsError('Preserve fresh cases and future human votes')
    cases=fresh_cases();known=set()
    paths=[ROOT/p for p in ['data/dev/train.jsonl','data/dev/validation.jsonl','data/training_expansion/blanket600_v1/expanded.jsonl','data/hybrid_inference_v1/inputs.json','data/reward_diagnostics/onpolicy_v1/scenarios.jsonl','data/reward_diagnostics/onpolicy_v2/scenarios.jsonl','data/training_preferences/human_dpo_v1/fresh_audit.jsonl']]
    for path in paths:
        records=json.loads(path.read_text()) if path.suffix=='.json' else [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        for r in records:
            message=r['message'];known.add(normalize(message['text'] if isinstance(message,dict) else message))
    if any(normalize(c['message']) in known for c in cases):raise ValueError('Incoming message overlaps previous inspected/training cases')
    assert len({normalize(c['message']) for c in cases})==30
    folder.mkdir(parents=True)
    (folder/'inputs.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(cases=30,categories={c:sum(r['category']==c for r in cases) for c in sorted({r['category'] for r in cases})},scope='Fresh assistant-authored conversational inputs, no reference replies or training labels. Same domains as prior work; not independent population sample.',training=False,locked_test_read=False,overlap_check='Incoming messages normalized and checked against explicit train/validation/expanded and prior diagnostic inputs; no locked test opened',input_sha256=hashlib.sha256((folder/'inputs.json').read_bytes()).hexdigest())
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
