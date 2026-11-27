"""Pure data boundaries for the user-calibrated SEA-LION restart."""
import math


def conversation(row):
    if len(row['context']) != len(row['speakers']):
        raise ValueError('Speaker coverage mismatch')
    names = {'user': 'You', 'peer': 'Friend', 'other': 'Other'}
    return '\n'.join([*(names[s]+': '+t for s,t in zip(row['speakers'],row['context'],strict=True)),
                       'Friend: '+row['message']])


def prompt(row, style, anchor=None):
    text = conversation(row)
    if style == 'LOCAL_REWRITE':
        if not isinstance(anchor,str) or not anchor.strip():
            raise ValueError('A nonempty actual English anchor is required')
        return ('Rewrite the English draft in natural Singlish. Preserve its people, facts, timing and decision.\n'
                +text+'\nEnglish draft: '+anchor.strip()+'\nYou (LOCAL_REWRITE):')
    if style not in {'STANDARD','LOCAL'} or anchor is not None:
        raise ValueError('Invalid direct reply route')
    return text+'\nYou ('+style+'):'


def from_record(record):
    return dict(item_id=record['id'],context=[t['text'] for t in record['session_memory']],
                speakers=[t['speaker'] for t in record['session_memory']],message=record['message']['text'],
                standard=record['replies']['standard'],local=record['replies']['local'])


def filter_training(records, excluded):
    ids=[r['id'] for r in records]
    if len(set(ids)) != len(ids) or not set(excluded) <= set(ids):
        raise ValueError('Duplicate IDs or unknown exclusion')
    return [from_record(r) for r in records if r['id'] not in excluded]


def check_splits(groups):
    seen=set()
    for name,rows in groups.items():
        ids={r['item_id'] for r in rows}
        if len(ids)!=len(rows) or ids & seen:
            raise ValueError('Duplicate/overlapping split '+name)
        seen |= ids
    # Exact conversation leakage is checked independently of IDs and targets.
    inputs=set()
    for name,rows in groups.items():
        text={conversation(r) for r in rows}
        if len(text)!=len(rows) or text & inputs:
            raise ValueError('Repeated conversation across split '+name)
        inputs |= text


def selection(old, new):
    if not all(math.isfinite(x) and x >= 0 for x in [old,new]):
        raise ValueError('Invalid validation loss')
    return .5*old+.5*new
