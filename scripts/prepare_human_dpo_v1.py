"""Export explicit human wins, deduplicate and split by conversation."""
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def prepare(source,items_path,output):
    if output.exists():raise FileExistsError('Preserve frozen preference datasets')
    packet=json.loads((source/'packet.json').read_text());key=json.loads((source/'key.json').read_text());votes=json.loads((source/'ratings.json').read_text())
    assert hashlib.sha256((source/'packet.json').read_bytes()).hexdigest()==key['packet_sha256']
    questions={q['id']:q for q in packet['questions']}
    if not set(votes)<=set(questions):raise ValueError('Unknown vote IDs')
    pairs={};excluded={'tie':0,'both_bad':0,'identical':0};directions={}
    for qid,v in votes.items():
        q=questions[qid];choice=v['choice']
        if choice not in ('A','B'):
            if choice=='Tie':excluded['tie']+=1
            elif choice=='Both bad':excluded['both_bad']+=1
            else:raise ValueError('Invalid vote')
            continue
        chosen=q[choice].strip();rejected=q['B' if choice=='A' else 'A'].strip()
        if chosen==rejected:excluded['identical']+=1;continue
        signature=(q['item_id'],chosen,rejected)
        directions.setdefault((q['item_id'],*sorted((chosen,rejected))),set()).add((chosen,rejected))
        pairs.setdefault(signature,dict(item_id=q['item_id'],context=[t['text'] for t in q['conversation']],speakers=[t['speaker'] for t in q['conversation']],message=q['message'],chosen=chosen,rejected=rejected,review_ids=[]))['review_ids'].append(qid)
    conflicts={k for k,v in directions.items() if len(v)>1}
    rows=[r for r in pairs.values() if (r['item_id'],*sorted((r['chosen'],r['rejected']))) not in conflicts]
    ids=sorted({r['item_id'] for r in rows});random.Random(425).shuffle(ids)
    n=max(1,round(len(ids)*.2));held=set(ids[:n]);train=set(ids[n:])
    if not train:raise ValueError('Need multiple conversations for grouped split')
    output.mkdir(parents=True)
    for name,group in [('train',train),('validation',held)]:
        (output/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if r['item_id'] in group))
    items=[json.loads(l) for l in items_path.read_text().splitlines() if l.strip()]
    audit=[i for i in items if i['id'] not in train]
    (output/'audit_conversations.jsonl').write_text(''.join(json.dumps(i,ensure_ascii=False)+'\n' for i in audit))
    manifest=dict(seed=425,raw_decisive=sum(v['choice'] in ('A','B') for v in votes.values()),unique_pairs=len(rows),deduplicated=sum(len(r['review_ids'])-1 for r in rows),conflicting_unordered_pairs=len(conflicts),excluded=excluded,train_conversations=len(train),validation_conversations=len(held),train_pairs=sum(r['item_id'] in train for r in rows),validation_pairs=sum(r['item_id'] in held for r in rows),train_ids=sorted(train),validation_ids=sorted(held),source_sha256={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in ('packet.json','key.json','ratings.json')},limitation='Former development conversations explicitly repurposed for human preference training. Grouped held-out preferences are unseen by DPO updates but previously used in SFT selection and human review; not a pristine test. No locked test read. Ties/both-bad never converted to positives; written notes not silently treated as corrected gold.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

if __name__=='__main__':
    print(json.dumps(prepare(ROOT/'data/human_review/model_ab_v1',ROOT/'data/dev/validation.jsonl',ROOT/'data/training_preferences/human_dpo_v1'),indent=2))
