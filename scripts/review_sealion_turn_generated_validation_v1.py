"""Generate anonymous validation options for all frozen English snapshots."""
import json
from pathlib import Path
import random
import re
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_turn_repair_v1 as study
import run_sealion_turn_layout_v2 as layout


def main():
    assert Path.home().name=='hm.teoh.2024'
    study.torch.set_num_threads(4)
    assert study.read('results.json')['status']=='complete'
    assert study.read('layout_v2/results.json')['status']=='complete'
    criteria=dict(primary='Count of16 generated validation English replies addressing the latest message and preserving explicit facts/actors. Use pass/fail/uncertain; wording need not match references.',
        retention='Before proposing any candidate, check20 reused legacy English generations; no clear aggregate drop allowed. No automatic default promotion.',
        tie_break='Fewer hard role/fact contradictions, then smaller uncertainty count. If no clear gain, retain original.',
        scope='Author-written synthetic validation, assistant review, not independent human accuracy. Original teacher-forced retention selection remains recorded separately. Challenge/user outputs cannot choose a candidate.',
        training_performed=False)
    study.save('review/generated_validation_criteria.json',criteria)
    rows=study.read('data/validation.json')
    model=study.load()
    tok=study.checkpoint_tokenizer(study.BUNDLE/'base')
    canonical_source=study.source;canonical_boundary=study.BOUNDARY
    routes=[('original','default','canonical'),('untrained_layout','default','layout')]
    for prefix,arms in [('', ['supervised','contrastive']),('layout_v2/', ['supervised'])]:
        for arm in arms:
            for step in [40,80,120]:
                name=('layout_' if prefix else '')+arm+str(step)
                model.load_adapter(ROOT/prefix/f'outputs/{arm}/step_{step:04d}',adapter_name=name,is_trainable=False)
                routes.append((name,name,'layout' if prefix else 'canonical'))
    reports={}
    for name,adapter,format_name in routes:
        study.activate(model,adapter)
        study.source=layout.marked_source if format_name=='layout' else canonical_source
        study.BOUNDARY=re.compile(r'\n(?:(?:Me|Friend|Other)(?: \([A-Z_]+\))?|Earlier conversation|Latest incoming message from Friend):') if format_name=='layout' else canonical_boundary
        outputs=[]
        for row in rows:
            out=study.decode(model,tok,row)
            outputs.append(dict(item_id=row['item_id'],english=out['reply'],status=out['status']))
        reports[name]=dict(source_format=format_name,outputs=outputs)
        study.save('review/generated_validation_raw.json',reports)
        print('Generated validation',name,len(outputs),flush=True)
    rng=random.Random(841106)
    packet=[];key={}
    for i,row in enumerate(rows):
        unique={}
        for name,report in reports.items():unique.setdefault(report['outputs'][i]['english'],[]).append(name)
        texts=list(unique);rng.shuffle(texts)
        options={chr(65+j):text for j,text in enumerate(texts)}
        packet.append(dict(item_id=row['item_id'],context=list(zip(row['speakers'],row['context'],strict=True)),
            message=row['message'],reference=row['positive'],options=options))
        key[row['item_id']]={letter:unique[text] for letter,text in options.items()}
    study.save('review/generated_validation_packet.json',dict(criteria=criteria,cases=packet))
    study.save('review/generated_validation_key.json',key)
    print('COMPLETE generated validation',len(routes),'routes',flush=True)


if __name__=='__main__':main()
