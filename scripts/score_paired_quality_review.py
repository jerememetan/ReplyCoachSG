"""Unblind a complete paired review; never substitute reference-token loss for quality."""
import argparse,json
from pathlib import Path
from hashlib import sha256
from collections import Counter

AXES=('context','same_meaning','singlish')

def score(packet_path,key_path,ratings_path):
    packet=json.loads(packet_path.read_text());key=json.loads(key_path.read_text());ratings=json.loads(ratings_path.read_text())
    digest=sha256(packet_path.read_bytes()).hexdigest()
    if digest!=key['packet_sha256'] or digest!=ratings['packet_sha256']:raise ValueError('Packet hash mismatch')
    expected={(q['n'],letter) for q in packet['questions'] for letter in q['options']}
    rows=ratings['ratings']
    if len(rows)!=len(expected) or {(r['n'],r['option']) for r in rows}!=expected:raise ValueError('Every whole pair must be rated once')
    counts={}
    for r in rows:
        if not r.get('reason') or any(r.get(a) not in ('pass','fail','uncertain') for a in AXES):raise ValueError('Explicit axis judgments and reason required')
        arm=key['mapping'][str(r['n'])][r['option']];c=counts.setdefault(arm,Counter());c['pairs']+=1
        for a in AXES:c[a+'_'+r[a]]+=1
        c['all_three_pass']+=all(r[a]=='pass' for a in AXES)
    return dict(checkpoint=packet['checkpoint'],seed=packet.get('seed',425),counts={a:dict(c) for a,c in counts.items()},
        packet_sha256=digest,ratings_sha256=sha256(ratings_path.read_bytes()).hexdigest(),
        limitation=ratings.get('limitation','')+' Assistant judgments are not independent human gold; 50 reused development conversations.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','key','ratings','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh output required')
    result=score(a.packet,a.key,a.ratings);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
