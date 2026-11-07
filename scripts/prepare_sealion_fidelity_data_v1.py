"""Assistant-authored controlled fidelity pairs; splits hold templates/values apart."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/training_expansion/sealion_fidelity_v1'

def make(split):
    records=[]
    times={'train':['06:15','06:45','07:20','07:50','08:05','08:40','09:10','09:35','10:25','11:55','13:30','16:45'],'validation':['07:25','08:50','09:45','12:20'],'challenge':['06:35','08:10','10:40','14:25']}[split]
    if split=='train':
        templates=[('I cannot leave before {v}.','Before {v} cannot leave leh.','Can you meet earlier?'),('Please return it before {v}.','Return before {v} ah.','When do you need it back?'),('I will arrive at {v}.','I reach at {v}.','What time will you arrive?'),('Could we meet after {v}?','After {v} meet can?','When shall we meet?')]
    elif split=='validation':templates=[('I need to stay here until {v}.','I need to stay here until {v} leh.','Can you leave now?'),('We agreed on {v}.','We said {v} lah.','What time did we agree on?')]
    else:templates=[('I cannot finish work before {v}.','Before {v} cannot finish work leh.','Can we meet before you finish work?'),('Please have it back to me by {v}.','By {v} return me ah.','When should I return it?')]
    for v in times:
        for en,sg,msg in templates:
            records.append(dict(kind='time',value=v,standard_anchor=en.format(v=v),response=sg.format(v=v),context=[],speakers=[],message=msg))
    objects={'train':['charger','adapter','notebook','umbrella'],'validation':['camera','helmet'],'challenge':['goggles','raincoat']}[split]
    dates=['Monday','Tuesday','Thursday','Friday'] if split=='train' else ['Saturday','Sunday']
    for obj in objects:
        for date in dates:
            for direction in ['return_to_me','return_to_you']:
                if split=='train':
                    if direction=='return_to_me':en=f'Please return my {obj} on {date}.';sg=f'{date} return my {obj} ah.'
                    else:en=f'I will return your {obj} on {date}.';sg=f'{date} I return your {obj}.'
                elif split=='validation':
                    if direction=='return_to_me':en=f'Can you bring my {obj} back on {date}?';sg=f'{date} bring my {obj} back can?'
                    else:en=f'I can bring your {obj} back on {date}.';sg=f'{date} I bring your {obj} back.'
                else:
                    if direction=='return_to_me':en=f'Yes please, give me back my {obj} on {date}.';sg=f'Yes please, {date} return me my {obj}.'
                    else:en=f'I still have your {obj}; I will give it back on {date}.';sg=f'Your {obj} still with me. {date} I return you.'
                context=[f'You still have my {obj}.'] if direction=='return_to_me' else [f'I still have your {obj}.']
                msg=f'Can I return it on {date}?' if direction=='return_to_me' else f'Can you return it on {date}?'
                records.append(dict(kind='ownership',direction=direction,object=obj,value=date,standard_anchor=en,response=sg,context=context,speakers=['user'],message=msg))
    amounts={'train':['six','eight','eleven','fourteen','seventeen','twenty'],'validation':['nine','thirteen'],'challenge':['seven','twelve']}[split]
    for v in amounts:
        if split=='train':en=f'My budget is {v} dollars. Could we find something cheaper?';sg=f'My budget {v} dollars only leh. Find cheaper one can?'
        elif split=='validation':en=f'I can spend no more than {v} dollars.';sg=f'I can spend at most {v} dollars leh.'
        else:en=f'I only have {v} dollars available for lunch.';sg=f'Lunch budget {v} dollars only leh.'
        records.append(dict(kind='amount',value=v,standard_anchor=en,response=sg,context=[],speakers=[],message='Would you like the expensive lunch set?'))
    for i,r in enumerate(records,1):r.update(item_id=f'FIDELITY-{split.upper()}-{i:03d}',provenance='Assistant-authored controlled template pair, not human gold; supplemental experiment only. No team record edited.')
    return records

def main():
    if OUT.exists():raise FileExistsError('Fresh version required')
    OUT.mkdir(parents=True)
    groups={s:make(s) for s in ['train','validation','challenge']}
    signatures={s:{(tuple(r['context']),r['message'],r['standard_anchor']) for r in rs} for s,rs in groups.items()}
    assert not signatures['train']&(signatures['validation']|signatures['challenge']) and not signatures['validation']&signatures['challenge']
    for s,rs in groups.items():(OUT/f'{s}.json').write_text(json.dumps(rs,ensure_ascii=False,indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(counts={s:len(rs) for s,rs in groups.items()},scope='Controlled supplemental template test; new input/anchor strings and template variants split apart. Objects/times may occur in historical training or pretraining; not independent unseen-domain gold. Challenge has no training gradients or checkpoint selection.',human_gold=False,team_data_changed=False,training_schedule='745 audited rewrite replay pairs +745 targeted exposures/epoch; smaller unique targeted set cycled evenly.'),indent=2)+'\n')
    print({s:len(rs) for s,rs in groups.items()})
if __name__=='__main__':main()
