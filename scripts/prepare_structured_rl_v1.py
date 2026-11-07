import hashlib,json
from pathlib import Path
from replycoach.generate.structured_rl_v1 import make_cases,reward
folder=Path('data/training_diagnostics/structured_rl_v1')
if folder.exists():raise FileExistsError('Preserve frozen benchmark')
folder.mkdir(parents=True)
cases=make_cases()
for split in ('train','validation','test'):
    path=folder/f'{split}.jsonl';path.write_text(''.join(json.dumps(c)+'\n' for c in cases if c['split']==split))
manifest=dict(task='Choose one of four explicit structured actions satisfying stated facts; exactly one feasible option',splits=dict(train=192,validation=48,test=48),test_scope='New synthetic benchmark test, not locked product test',labels='Deterministic verifier, not human conversation-quality labels',model_inputs='Only task instruction, facts and four actions; no split, ID, reward or correct index',universal_refusal='Refusal is not an action in this narrow task. This does not redefine the user policy that refusals can be good conversational replies.',limitations='Shared task families across splits; disjoint fact states/objects, not broad language generalization. Success does not establish better free-form STANDARD or LOCAL replies.',files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.jsonl')})
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Frozen192 train,48 validation,48 synthetic test cases.')
