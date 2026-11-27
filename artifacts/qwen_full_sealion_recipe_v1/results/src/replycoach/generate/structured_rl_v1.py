"""Mechanically verified contextual-bandit benchmark; not free-form chat scoring."""
import itertools,json,random,re
from pathlib import Path

DAYS=('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')
LETTERS=('A','B','C','D')

def minute(text):
    if not isinstance(text,str) or not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d',text):raise ValueError('Invalid24-hour clock')
    hour,mins=map(int,text.split(':'));return hour*60+mins

def clock(value):return f'{value//60:02}:{value%60:02}'

def reward(case,action):
    """+1 exactly when action satisfies explicitly defined structured task; else -1."""
    if not isinstance(action,dict):return -1
    f=case['facts'];category=case['category']
    if category=='time':
        if set(action)!={'action','at'} or action['action']!='meet':return -1
        try:return 1 if minute(f['earliest'])<=minute(action['at'])<=minute(f['latest']) else -1
        except (ValueError,TypeError):return -1
    if category=='availability':
        return 1 if set(action)=={'action','day'} and action['action']=='meet' and action['day'] in f['available_days'] else -1
    if category=='ownership':
        return 1 if set(action)=={'action','actor','recipient','item'} and action['action']=='return' and action['actor']==f['holder'] and action['recipient']==f['owner'] and action['item']==f['item'] and f['owner']!=f['holder'] else -1
    raise ValueError('Unknown structured task')

def prompt(case):
    instructions={'time':'Choose a meeting time within the inclusive earliest/latest bounds.', 'availability':'Choose a meeting day listed as available.', 'ownership':'Choose the return action that moves the item from its holder to its owner.'}
    text=instructions[case['category']]+' Output only the option letter.\nFacts: '+json.dumps(case['facts'],sort_keys=True)
    for letter,action in zip(LETTERS,case['actions']):text+='\n'+letter+': '+json.dumps(action,sort_keys=True)
    return text

def make_cases(seed=425):
    rng=random.Random(seed);cases=[]
    daysets=[list(d) for n in range(1,5) for d in itertools.combinations(DAYS,n)]
    rng.shuffle(daysets)
    windows=set()
    objects=[a+' '+b for a in ('red','blue','green','yellow','black','white','orange','purple','silver','gold','pink','brown') for b in ('umbrella','notebook','jacket','racket','cable','camera','charger','bicycle')]
    rng.shuffle(objects)
    for split,count in [('train',64),('validation',16),('test',16)]:
        positions={c:[i%4 for i in range(count)] for c in ('time','availability','ownership')}
        for order in positions.values():rng.shuffle(order)
        owners=['user','peer']*(count//2);rng.shuffle(owners)
        for n in range(count):
            for category in ('time','availability','ownership'):
                number=sum(c['category']==category for c in cases)
                if category=='time':
                    while True:
                        start=rng.randrange(8*60,18*60,5);end=start+rng.randrange(30,125,5)
                        if (start,end) not in windows:windows.add((start,end));break
                    facts=dict(earliest=clock(start),latest=clock(end))
                    good=dict(action='meet',at=clock(rng.randrange(start,end+1,5)))
                    bad=[dict(action='meet',at=clock(t)) for t in (start-10,end+10,end+25)]
                elif category=='availability':
                    available=daysets[number];facts=dict(available_days=available)
                    good=dict(action='meet',day=rng.choice(available))
                    bad=[dict(action='meet',day=d) for d in rng.sample([d for d in DAYS if d not in available],3)]
                else:
                    owner=owners[n];holder='peer' if owner=='user' else 'user'
                    facts=dict(owner=owner,holder=holder,item=objects[number])
                    good=dict(action='return',actor=holder,recipient=owner,item=facts['item'])
                    bad=[dict(action='return',actor=a,recipient=b,item=facts['item']) for a,b in itertools.product(('user','peer'),repeat=2) if (a,b)!=(holder,owner)]
                rng.shuffle(bad);actions=bad[:];actions.insert(positions[category][n],good)
                case=dict(id=f'SRL-{split}-{category}-{n:03}',split=split,category=category,facts=facts,actions=actions)
                assert sum(reward(case,a)==1 for a in actions)==1
                cases.append(case)
    return cases


def encode_cases(cases,tokenizer,max_source=384):
    encoded=[]
    for case in cases:
        source=tokenizer(prompt(case))['input_ids']
        if len(source)>max_source:raise ValueError('Refuse structured source truncation')
        encoded.append([(source,tokenizer(letter)['input_ids']) for letter in LETTERS])
    return encoded


def action_distribution(scores,temperature=2.,exploration=.1):
    return (1-exploration)*(scores/temperature).softmax(-1)+exploration/scores.shape[-1]
