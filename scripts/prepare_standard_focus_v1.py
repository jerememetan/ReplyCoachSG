"""Frozen assistant-authored role/intent contrasts; development only, not human gold."""
import copy, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/synthetic/standard_focus_v1'

def build():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    original=[json.loads(l) for l in (ROOT/'data/dev/train.jsonl').read_text().splitlines()]
    targets={x['item_id']:x['replies'] for x in json.loads((ROOT/'data/distillation/shared_intent_v1_reviewed.json').read_text())['items']}
    control=copy.deepcopy(original)
    for x in control:x['replies']=targets[x['id']]
    additions=[]
    def pair(family,variant,context,message,standard,local):
        additions.append(dict(id=f'SYN-FOCUS-{family}-{variant:02d}',split='dev',session_memory=[dict(speaker=s,text=t) for s,t in context],message=dict(text=message),replies=dict(standard=standard,local=local),provenance=dict(author='Codex assistant',kind='hand-authored contrast',family=family,not_human_gold=True)))
    for i,obj in enumerate(['calculator','umbrella','power bank','adapter']):
        pair('lend',i, [('user',f'I have my {obj} with me.')],f'Could I use your {obj} for a while?',f'Sure, you can use my {obj}.',f'Can, my {obj} you use first.')
        pair('borrow',i,[('user',f'I forgot my {obj}.')],f'Would you like to use my {obj}?',f'Yes please, could you lend me your {obj}?',f'Want, can lend me your {obj} ah?')
    for i,obj in enumerate(['workshop notes','practice questions','project outline','event photos']):
        pair('send',i,[('user',f'I have the {obj} saved.')],f'Could you share the {obj} with me?',f'Sure, I will send you the {obj}.',f'Can, I send you the {obj}.')
        pair('receive',i,[('user',f'I cannot find the {obj}.')],f'I have the {obj}. Want me to send them?',f'Yes please, send me the {obj}.',f'Want, send me the {obj} can?')
    for i,obj in enumerate(['scanner','binding machine','studio microphone','ticket kiosk']):
        pair('neither_knows',i,[('user',f'I do not know how to use the {obj}.')],f'I have never used the {obj} either.',f'Let us ask someone else how to use the {obj}.',f'Then we ask someone else how to use the {obj} lah.')
        pair('peer_knows',i,[('user',f'I do not know how to use the {obj}.')],f'I know how to use the {obj}.',f'Could you show me how to use the {obj}?',f'Can show me how to use the {obj} anot?')
    for i,event in enumerate(['board game session','study meetup','museum trip','group lunch']):
        pair('decline',i,[('user',f'Would you like to join our {event}?')],f'I cannot make it to the {event}; I have work.', 'No worries, we can meet another time.', 'Never mind lah, next time then meet.')
        pair('accept',i,[('user',f'Would you like to join our {event}?')],f'I can make it to the {event}.', 'Great, see you there!', 'Swee, see you there ah!')
    for i,(early,late) in enumerate([('two','four'),('three','five'),('six','eight'),('seven','nine')]):
        pair('later',i,[('user',f'I cannot meet before {late}.')],f'Can we meet at {early}?',f'I cannot make {early}. Could we meet at {late} instead?',f'{early.capitalize()} cannot leh. {late.capitalize()} instead can?')
        pair('earlier',i,[('user',f'I need to leave by {early}.')],f'Can we meet at {late}?',f'I have to leave by {early}. Could we meet before then?',f'I {early} must leave liao. Before that meet can?')
    for i,obj in enumerate(['poster','lab report','presentation','reading summary']):
        pair('unfinished',i,[('user',f'I have not finished my {obj}.')],f'Is your {obj} ready to send?',f'Not yet, I will send my {obj} when it is ready.',f'Haven’t finish my {obj} leh. Ready then send you.')
        pair('finished',i,[('user',f'I have finished my {obj}.')],f'Is your {obj} ready to send?',f'Yes, my {obj} is ready. I will send it now.',f'My {obj} done liao, now send you.')
    for i,place in enumerate(['seminar room','art studio','rehearsal space','computer lab']):
        pair('where',i,[('peer',f'The {place} has moved to level three.')],f'Which level is the {place} on now?',f'The {place} is on level three now.',f'{place.capitalize()} now level three lah.')
        pair('unknown_where',i,[('peer',f'The {place} has moved, but I do not know where.')],f'Do you know where the {place} is now?',f'I am not sure. Let us check where the {place} moved.',f'Not sure leh, we check {place} move where liao.')
    for i,event in enumerate(['revision class','team briefing','training session','committee meeting']):
        pair('update_peer',i,[('user',f'I will attend the {event} and can take notes for you.')],f'Could you tell me what happens at the {event}?',f'Sure, I will update you after the {event}.',f'Can, {event} finish then I update you.')
        pair('ask_update',i,[('user',f'I will miss the {event}.'),('peer','I will be there and can take notes.')],f'Would you like an update after the {event}?',f'Yes please, let me know what I miss at the {event}.',f'Want, {event} I miss what then tell me ah.')
    for i,obj in enumerate(['camera','tripod','textbook','racket']):
        pair('return_mine',i,[('peer',f'I still have your {obj}.')],f'When should I return your {obj}?',f'Please return my {obj} tomorrow.',f'My {obj} tomorrow return me can?')
        pair('return_yours',i,[('user',f'I still have your {obj}.')],f'Could you return my {obj} tomorrow?',f'Sure, I will return your {obj} tomorrow.',f'Can, tomorrow return your {obj}.')
    for i,(food,alternative) in enumerate([('spicy noodles','plain rice'),('peanut cookies','fruit'),('milk tea','water'),('prawn noodles','vegetable soup')]):
        pair('food_no',i,[('user',f'I cannot have {food}.')],f'Should I order {food} for you?',f'No thanks, I cannot have {food}. Could I have {alternative} instead?',f'{food.capitalize()} cannot leh. {alternative.capitalize()} instead can?')
        pair('food_yes',i,[('user',f'I would like {food}.')],f'Should I order {food} for you?',f'Yes please, order {food} for me.',f'Want, help me order {food} ah.')
    # Auxiliary context rows remain original-only: none of these new contexts has a peer->user transition.
    for name,rows in [('control',control),('targeted',control+additions),('additions',additions)]:
        (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    print(json.dumps(dict(original=len(control),new=len(additions),total=len(control+additions))))
if __name__=='__main__':build()
