"""Prepare anonymous joint-draft review, then score complete explicit axis ratings."""
import argparse,hashlib,json,random,secrets
from pathlib import Path
ARMS=('control','targeted','capacity','standard_heavy')
AXES=('standard','local','same_meaning','singlish')
def prepare(directory,output,arms=ARMS,seed=425):
    if output.exists():raise FileExistsError(output)
    rng=random.Random(secrets.randbits(64));questions=[];mapping={};hashes={}
    for split,path in [('validation',Path('data/dev/validation.jsonl')),('diagnostic',Path('data/reward_diagnostics/standard_focus_v1.jsonl'))]:
        items=[json.loads(l) for l in path.read_text().splitlines()]
        reports={a:json.loads((directory/f'{a}_s{seed}{ "_diagnostic" if split=="diagnostic" else ""}.json').read_text()) for a in arms}
        for a,r in reports.items():
            if split=='validation':assert r['optimizer_steps']==784
            hashes[f'{split}/{a}']=hashlib.sha256((directory/f'{a}_s{seed}{ "_diagnostic" if split=="diagnostic" else ""}.json').read_bytes()).hexdigest()
        runs={a:{r['item_id']:r for r in report['outputs' if split=='diagnostic' else 'final_validation_outputs']} for a,report in reports.items()}
        assert all(set(run)=={c['id'] for c in items} for run in runs.values())
        for c in sorted(items,key=lambda x:x['id']):
            n=len(questions)+1;order=list(arms);rng.shuffle(order);options={};mapping[str(n)]={}
            for letter,arm in zip('ABCD',order):
                row=runs[arm][c['id']];options[letter]=dict(standard=row['generated_standard'],local=row['generated_local']);mapping[str(n)][letter]=arm
            questions.append(dict(n=n,split=split,item_id=c['id'],conversation=[[t['speaker'],t['text']] for t in c['session_memory']],message=c['message']['text'],options=options))
    output.mkdir(parents=True)
    packet=dict(seed=seed,rubric=dict(standard='Coherent next user turn; valid alternatives, refusals and plausible new selectable facts allowed; explicit contradictions and reversed roles fail.',local='Same contextual adequacy criterion, judged separately.',same_meaning='Preserve essential actors, facts, requests, decisions and commitments.',singlish='Recognizably natural Singlish; neutral English copies fail, uncertain naturalness may abstain.'),questions=questions)
    public=output/'packet.json';public.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    (output/'key.json').write_text(json.dumps(dict(mapping=mapping,packet_sha256=hashlib.sha256(public.read_bytes()).hexdigest(),source_sha256=hashes),indent=2)+'\n')
def score(output):
    from collections import Counter
    packet_path=output/'packet.json';packet=json.loads(packet_path.read_text());key=json.loads((output/'key.json').read_text());ratings=json.loads((output/'ratings.json').read_text())
    assert key['packet_sha256']==ratings['packet_sha256']==hashlib.sha256(packet_path.read_bytes()).hexdigest()
    expected={(q['n'],o) for q in packet['questions'] for o in q['options']}
    assert len(ratings['ratings'])==len(expected) and {(r['n'],r['option']) for r in ratings['ratings']}==expected
    splits={q['n']:q['split'] for q in packet['questions']};counts={}
    for r in ratings['ratings']:
        assert all(r.get(a) in ('pass','fail','uncertain') for a in AXES) and r.get('reason')
        arm=key['mapping'][str(r['n'])][r['option']];counter=counts.setdefault(splits[r['n']],{}).setdefault(arm,Counter());counter['pairs']+=1
        for a in AXES:counter[a+'_'+r[a]]+=1
        counter['all_pass']+=all(r[a]=='pass' for a in AXES)
    result=dict(seed=packet.get('seed',425),counts=counts,packet_sha256=key['packet_sha256'],ratings_sha256=hashlib.sha256((output/'ratings.json').read_bytes()).hexdigest(),limitation='One seed; assistant-authored training/diagnostics and assistant judgments; not independent human gold. Same50 reused validation conversations and24 fresh correlated development diagnostics. No final-test claims.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','score']);p.add_argument('--seed',type=int,default=425);p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS));p.add_argument('--directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.action=='prepare':prepare(a.directory,a.output,a.arms,a.seed)
    else:score(a.output)
