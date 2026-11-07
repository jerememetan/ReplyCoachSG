"""Tiny training-fit diagnostic; same-scenario reorderings are not independent tests."""
import copy,hashlib,json
from pathlib import Path
from replycoach.generate.structured_rl_v1 import reward
ROOT=Path(__file__).resolve().parents[1]

def pick(rows):
    selected=[]
    for category in ('time','availability','ownership'):
        for position in range(4):
            selected.append(next(c for c in rows if c['category']==category and reward(c,c['actions'][position])==1))
    return selected


def rotate(case,shift):
    row=copy.deepcopy(case);row['source_id']=case['id'];row['id']=case['id']+f'-rotate{shift}'
    row['actions']=row['actions'][shift:]+row['actions'][:shift]
    return row


def prepare(folder=ROOT/'data/training_diagnostics/structured_fit_v1'):
    if folder.exists():raise FileExistsError('Preserve frozen diagnostic')
    source=ROOT/'data/training_diagnostics/structured_rl_v1'
    train=pick([json.loads(l) for l in (source/'train.jsonl').read_text().splitlines()])
    novel=pick([json.loads(l) for l in (source/'validation.jsonl').read_text().splitlines()])
    perms=[rotate(c,n) for c in train for n in (1,2,3)]
    folder.mkdir(parents=True)
    for name,rows in [('train',train),('permuted',perms),('novel',novel)]:
        (folder/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest=dict(train_cases=12,permuted_cases=36,novel_cases=12,purpose='Can existing model/input/objective fit tiny training data?',permuted_scope='Same12 facts, all three nonidentity cyclic option rotations; no independent scenario claim',novel_scope='Twelve previously reused synthetic validation cases, no gradient or checkpoint selection; not untouched test',training_selection='One case per correct option position per category, frozen before diagnostic results',source_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('train.jsonl','validation.jsonl')},no_locked_or_synthetic_test_read=True)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
