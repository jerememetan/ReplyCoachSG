"""Prepare anonymous validation reply pairs, keeping arm keys in separate files."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random,secrets


def prepare(directory,arms,seeds,items_path,output):
    items={d['id']:d for d in map(json.loads,items_path.read_text().splitlines())}
    if len(items)!=50 or any(d['split']!='dev' for d in items.values()):
        raise ValueError('Expected the frozen 50 development validation conversations')
    output.mkdir(exist_ok=True,parents=True)
    for seed in seeds:
        packet_path=output/f'packet_s{seed}.json'; key_path=output/f'key_s{seed}.json'
        if packet_path.exists() or key_path.exists(): raise FileExistsError('Do not overwrite review packets')
        paths=[directory/f'{arm}_s{seed}.json' for arm in arms]
        reports=[json.loads(p.read_text()) for p in paths]
        outputs=[{r['item_id']:r for r in report['validation_outputs']} for report in reports]
        if any(set(rows)!=set(items) for rows in outputs): raise ValueError('Different/incomplete validation outputs')
        rng_seed=secrets.randbits(64); rng=random.Random(rng_seed); questions=[]; mapping={}; identical=[]
        for item_id,item in sorted(items.items()):
            for style in ('STANDARD','LOCAL'):
                replies=[rows[item_id]['generated_'+style.lower()].strip() for rows in outputs]
                if replies[0]==replies[1]:
                    identical.append(dict(item_id=item_id,style=style,reply=replies[0])); continue
                flip=rng.random()<0.5; n=len(questions)+1
                questions.append(dict(n=n,item_id=item_id,style=style,
                    conversation=[[t['speaker'],t['text']] for t in item['session_memory']],
                    message=item['message']['text'],A=replies[int(flip)],B=replies[1-int(flip)]))
                mapping[str(n)]={'A':arms[int(flip)],'B':arms[1-int(flip)]}
        packet=dict(purpose='anonymized_assistant_validation_inspection_only',seed=seed,questions=questions,identical=identical,
                    limitation='Identical replies are not assumed acceptable; this packet contains no human gold labels.')
        packet_path.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
        key_path.write_text(json.dumps(dict(seed=seed,arms=arms,randomization_seed=rng_seed,
            packet_sha256=sha256(packet_path.read_bytes()).hexdigest(),
            source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in paths},mapping=mapping),indent=2)+'\n')
        print('Seed',seed,':',len(questions),'differing pairs;',len(identical),'identical')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path)
    p.add_argument('--arms',nargs=2,required=True); p.add_argument('--seeds',nargs='+',type=int,default=[425,1,2])
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    items=Path(__file__).resolve().parents[1]/'data/dev/validation.jsonl'
    prepare(args.directory,args.arms,args.seeds,items,args.output)


if __name__=='__main__': main()
