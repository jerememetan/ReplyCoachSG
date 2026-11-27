"""Personal-account, CPU-only exports/sweeps from saved candidate likelihoods."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import sealion_style_v1 as style


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.root
    assert Path.home().name=='hm.teoh.2024'
    report=json.loads((root/'outputs/algorithms/results.json').read_text())
    assert report['status']=='complete'
    assert report.get('source_bundle_weights_unchanged') is True
    rows=report['outputs'];lexicon=json.loads((root/'outputs/algorithms/style_lexicon.json').read_text())
    review=root/'review';review.mkdir(exist_ok=True)
    fields=['item_id','scope','english','context','message']+list(rows[0]['methods'])
    with (review/'algorithm_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow(dict(item_id=row['item_id'],scope=row['scope'],english=row['standard_anchor'],
                context=json.dumps(list(zip(row['speakers'],row['context'],strict=True)),ensure_ascii=False),
                message=row['message'],**{key:value['text'] for key,value in row['methods'].items()}))
    sweeps=[]
    for copy_weight in [0.,.25,.5,1.,2.]:
        for style_weight in [0.,.2]:
            counts=Counter();changed=[]
            for row in rows:
                valid,_=style.rank_candidates(row['standard_anchor'],row['candidate_pool'],lexicon,copy_weight,style_weight)
                text=valid[0]['text'] if valid else row['standard_anchor']
                if row['scope']=='reused_development':
                    counts['cases']+=1
                    counts['eligible']+=int(style.copy_eligible(row['standard_anchor']))
                    counts['unnecessary_copies']+=int(style.copy_features(row['standard_anchor'],text)['unnecessary_copy'])
                    counts['exact_copies']+=int(style.canonical(row['standard_anchor'])==style.canonical(text))
                if text!=row['methods']['beam4']['text']:changed.append(dict(item_id=row['item_id'],baseline=row['methods']['beam4']['text'],candidate=text))
            sweeps.append(dict(copy_weight=copy_weight,style_weight=style_weight,counts=dict(counts),changed=changed))
    save(review/'penalty_sweep.json',dict(scope='CPU rescoring of saved candidates; exploratory development trade-offs, not accuracy or a selected deployment policy.',sweeps=sweeps))
    desired=['SEADUAL-011','SEADUAL-015','SEADUAL-008','SEADUAL-010','SEADUAL-013']
    rng=random.Random(20261108)
    selected=[row for ident in desired for row in rows if row['item_id']==ident]
    remaining=[row for row in rows if row['scope']=='reused_development' and row['item_id'] not in desired and row['methods']['combined']['text']!=row['methods']['beam4']['text']]
    selected+=rng.sample(remaining,min(5,len(remaining)))
    selected+=[row for row in rows if row['scope']=='assistant_authored_diagnostic'][:5]
    packet=[];key={}
    for number,row in enumerate(selected,1):
        unique={}
        for method,value in row['methods'].items():unique.setdefault(value['text'],[]).append(method)
        texts=list(unique);rng.shuffle(texts)
        identity=f'STYLE-REVIEW-{number:03d}'
        options={chr(65+i):text for i,text in enumerate(texts)}
        packet.append(dict(review_id=identity,context=list(zip(row['speakers'],row['context'],strict=True)),
            message=row['message'],english=row['standard_anchor'],options=options,
            criteria=['Preserves people, facts, timing, polarity and decision','Natural restrained Singlish','Useful response in the conversation'],
            note='Unchanged wording may be appropriate. English draft can itself be wrong. No mandatory particles.'))
        key[identity]=dict(item_id=row['item_id'],scope=row['scope'],mapping={letter:unique[text] for letter,text in options.items()})
    if not (review/'blind_packet.json').exists():
        assert not (review/'blind_key.json').exists()
        save(review/'blind_packet.json',dict(seed=20261108,cases=packet,human_ratings=None))
        save(review/'blind_key.json',key)
    else:
        assert (review/'blind_key.json').exists(), 'Preserve the existing review packet and repair its key separately'
    summary=dict(algorithm_status=report['status'],cases=len(rows),metrics=report['metrics'],
        replay_differences=report.get('repair25_replay_differences',[]),source_bundle_weights_unchanged=report.get('source_bundle_weights_unchanged'),
        no_automatic_promotion=True,human_review_pending=True,
        review_cases=len(packet),analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(review/'summary.json',summary)
    print(json.dumps(dict(status='exported',cases=len(rows),review_cases=len(packet),review=str(review))),flush=True)


if __name__=='__main__':main()
