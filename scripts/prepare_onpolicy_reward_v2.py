"""Fresh paraphrased scenario instances; frozen before sampling and scoring."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'data/reward_diagnostics/onpolicy_v2'
scenarios=[
('time',['What time does your class finish?','It finishes at17:20. I cannot meet before17:20.'],['peer','user'],'Could we meet at16:45?'),
('time',['I have to leave the cafe by14:10 to catch my bus.'],['user'],'Can you stay at the cafe until14:45?'),
('time',['I need to be home no later than22:15.'],['user'],'Want to start a two-hour film at21:30?'),
('time',['The earliest I can arrive is08:40.'],['user'],'Are you able to get here at08:15?'),
('availability',['I cannot meet this Friday. I am free next Monday instead.'],['user'],'Can we do this Friday?'),
('availability',['Is the evening session okay?','No, I can attend only the lunchtime session.'],['peer','user'],'Come to the evening session with me?'),
('availability',['I have a shift for the whole of Sunday. Saturday is free.'],['user'],'Fancy a day trip on Sunday?'),
('availability',['I am away all next week. I will be back the following Monday.'],['user'],'Can we get coffee next Thursday?'),
('ownership',['You have my rain jacket from yesterday. I lent it to you.'],['user'],'Is it okay if I give your jacket back on Monday?'),
('ownership',['I borrowed your bicycle to get to class.'],['user'],'Please return my bicycle this afternoon.'),
('ownership',['Have you seen my keys?','Yes, your keys are in my backpack.'],['peer','user'],'Do you still have my keys with you?'),
('ownership',['You borrowed my charging cable for your trip.'],['user'],'Should I bring your cable back tonight?'),
]
if folder.exists():raise FileExistsError('Preserve frozen scenarios')
folder.mkdir(parents=True)
rows=[dict(item_id=f'ONPOLICY2-{n:03}',category=cat,context=context,speakers=speakers,message=message) for n,(cat,context,speakers,message) in enumerate(scenarios,1)]
(folder/'scenarios.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('Frozen12 fresh instances; four samples each.')
