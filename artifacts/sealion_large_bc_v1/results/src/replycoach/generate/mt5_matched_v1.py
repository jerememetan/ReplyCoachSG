"""Native mT5 serialization and loss for the matched SEA workflow experiments."""
import torch
from torch.nn import functional as F
from replycoach.generate.style_control import mt5_input

def source(row,style,anchor=None):
    return mt5_input('REPLY' if style=='DIALOGUE' else style,row['message'],row['context'],True,row.get('speakers'),standard_reply=anchor)

def encode(row,tok,mode='direct',max_source=1536,max_target=384):
    text=source(row,'LOCAL_REWRITE',row['standard_anchor']) if mode=='rewrite' else source(row,row.get('style','DIALOGUE'))
    ids=tok(text,add_special_tokens=True)['input_ids'];target=tok(row['response'],add_special_tokens=True)['input_ids']
    if not ids or not row['response'].strip():raise ValueError('Empty encoder/target')
    if len(ids)>max_source or len(target)>max_target:raise ValueError('No truncation: '+row['item_id'])
    return ids,target

def reply_loss(logits,labels):return F.cross_entropy(logits.float().transpose(1,2),labels,ignore_index=-100)

def task_rows(records):
    return [dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],style=s,response=r['replies'][s.lower()]) for r in records for s in ['STANDARD','LOCAL']]

def paired_rows(records,allowed):
    if not allowed<={r['id'] for r in records}:raise ValueError('Pair eligibility/source mismatch')
    return [dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],standard_anchor=r['replies']['standard'],response=r['replies']['local']) for r in records if r['id'] in allowed]
