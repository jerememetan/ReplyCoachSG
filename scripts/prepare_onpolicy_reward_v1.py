"""New scenario instances frozen before sampling; no target replies or adequacy labels."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'data/reward_diagnostics/onpolicy_v1'
scenarios=[
('time','I cannot meet before 18:45 tonight.','Would 17:30 work for dinner?'),
('time','I need to be home by 21:00 tonight.','Shall we start our two-hour movie at 20:00?'),
('time','My appointment is at 16:00 and cannot be moved.','Can you come for a meeting from 15:30 to 16:30?'),
('time','I can stay until 11:15, then I must leave.','Could you stay until noon?'),
('availability','I am unavailable on Wednesday. Thursday evening is free.','Can we meet on Wednesday evening?'),
('availability','I can attend the morning session, but I cannot attend the afternoon one.','Please join the afternoon session with us.'),
('availability','I am working all Saturday. I have Sunday off.','Want to go hiking on Saturday?'),
('availability','I am not free this weekend. I am free next Tuesday.','Shall we have lunch this Sunday?'),
('ownership','I lent you my badminton racket yesterday.','Can I return your racket next week instead?'),
('ownership','I borrowed your power bank for my trip.','Could you bring my power bank back tomorrow?'),
('ownership','I have your library card in my wallet.','Do you still have my library card?'),
('ownership','You borrowed my folding chair for your event.','Would you like me to return your chair this evening?'),
]
if folder.exists():raise FileExistsError('Preserve frozen sampling inputs')
folder.mkdir(parents=True)
rows=[dict(item_id=f'ONPOLICY-{n:03}',category=category,context=[context],speakers=['user'],message=message) for n,(category,context,message) in enumerate(scenarios,1)]
(folder/'scenarios.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('Frozen12 new scenarios, four samples each.')
