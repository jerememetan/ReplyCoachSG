"""Unblind a completed assistant review, verifying packet and rating provenance."""
import argparse,json
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict


def score(packet_path,key_path,ratings_path):
    packet=json.loads(packet_path.read_text()); key=json.loads(key_path.read_text()); ratings=json.loads(ratings_path.read_text())
    digest=sha256(packet_path.read_bytes()).hexdigest()
    if key['packet_sha256']!=digest or ratings['packet_sha256']!=digest:
        raise ValueError('Packet hash mismatch')
    qs={q['n']:q for q in packet['questions']}; rs=ratings['ratings']
    if len(rs)!=len(qs) or {r['n'] for r in rs}!=set(qs):
        raise ValueError('Every differing pair must be rated exactly once before unblinding')
    counts=Counter(); by_style=defaultdict(Counter)
    for r in rs:
        if r['choice'] not in ('A','B','tie','both_bad') or not r.get('reason'):
            raise ValueError('Invalid rating')
        outcome=key['mapping'][str(r['n'])][r['choice']] if r['choice'] in ('A','B') else r['choice']
        counts[outcome]+=1; by_style[qs[r['n']]['style']][outcome]+=1
    for r in packet['identical']: by_style[r['style']]['identical']+=1
    counts['identical']=len(packet['identical'])
    return dict(seed=packet['seed'],counts=dict(counts),by_style={k:dict(v) for k,v in by_style.items()},
                packet_sha256=digest,ratings_sha256=sha256(ratings_path.read_bytes()).hexdigest(),
                reviewer_limitations=ratings.get('limitation',''),
                limitation='Assistant inspection, not human gold. Seeds reuse the same conversations; do not count pooled cells as independent conversations.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','key','ratings','output'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Choose fresh output')
    result=score(args.packet,args.key,args.ratings); args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
