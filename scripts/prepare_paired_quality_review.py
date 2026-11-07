"""Hide model identities while reviewing whole STANDARD/LOCAL pairs, not isolated drafts."""
import argparse,json,random,secrets
from pathlib import Path
from hashlib import sha256

def prepare(directory,arms,output,validation,seed=425):
    if output.exists():raise FileExistsError('Fresh paired-review directory required')
    items={c['id']:c for c in map(json.loads,validation.read_text().splitlines())}
    if len(items)!=50:raise ValueError('Expected 50 frozen development validation cases')
    paths=[directory/f'{a}_s{seed}.json' for a in arms]
    runs=[json.loads(p.read_text()) for p in paths]
    outputs=[{c['item_id']:c for c in r['final_validation_outputs']} for r in runs]
    if any(set(o)!=set(items) for o in outputs):raise ValueError('Missing final-step outputs')
    if any(r['optimizer_steps']!=784 for r in runs):raise ValueError('Expected matched final step 784')
    randomization_seed=secrets.randbits(64);rng=random.Random(randomization_seed);questions=[];mapping={}
    for i,(item_id,c) in enumerate(sorted(items.items()),1):
        order=list(range(len(arms)));rng.shuffle(order)
        options={};mapping[str(i)]={}
        for letter,n in zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',order):
            row=outputs[n][item_id]
            options[letter]={'standard':row['generated_standard'],'local':row['generated_local']}
            mapping[str(i)][letter]=arms[n]
        questions.append(dict(n=i,item_id=item_id,conversation=[[t['speaker'],t['text']] for t in c['session_memory']],message=c['message']['text'],options=options))
    packet={'purpose':'paired assistant quality inspection, not human gold','checkpoint':'final step 784','seed':seed,
        'rubric':{'context':'Both replies work as the next user turn; valid alternatives to references allowed.',
                  'same_meaning':'Essential facts, decisions, requests and commitments compatible; minor politeness differences allowed.',
                  'singlish':'LOCAL is recognizably natural Singlish, not merely neutral English or particle stuffing.'},
        'questions':questions,'limitation':'Rate each axis pass/fail/uncertain. Model identities hidden but may be inferred. Same reused development conversations; no human accuracy claim.'}
    output.mkdir(parents=True)
    (output/'packet.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    (output/'key.json').write_text(json.dumps(dict(mapping=mapping,seed=seed,randomization_seed=randomization_seed,packet_sha256=sha256((output/'packet.json').read_bytes()).hexdigest(),source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
    print('Prepared',len(questions),'whole-pair cases,',len(arms),'arms; references and arm labels hidden.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--arms',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=425)
    args=p.parse_args();prepare(args.directory,args.arms,args.output,Path(__file__).resolve().parents[1]/'data/dev/validation.jsonl',args.seed)
