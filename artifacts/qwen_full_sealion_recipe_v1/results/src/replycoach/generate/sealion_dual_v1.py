"""Dual-style data, FP32 reply-only loss and legacy-cache generation."""
import re
import torch
from torch.nn import functional as F
from replycoach.generate.sealion_data import source

STYLES=('STANDARD','LOCAL')
BOUNDARY=re.compile(r'\n(?:Friend|Other|You(?: \((?:STANDARD|LOCAL)\))?):')


def item_rows(items):
    result=[]
    for r in items:
        if r['split']!='dev' or r.get('status')!='completed':raise ValueError('Only completed development records allowed')
        for style in STYLES:
            result.append(dict(item_id=r['id'],style=style,context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],response=r['replies'][style.lower()]))
    if len(result)!=2*len({r['item_id'] for r in result}):raise ValueError('Duplicate or missing style pairs')
    return result


def reply_loss(logits,labels):
    # The upstream causal objective, evaluated in FP32 for stable validation.
    return F.cross_entropy(logits[:,:-1,:].float().transpose(1,2),labels[:,1:],ignore_index=-100)


@torch.no_grad()
def generate_reply(model,tok,row,style,max_new_tokens=64,device='cuda'):
    if style not in STYLES:raise ValueError(style)
    model.eval()
    ids=tok.encode(source(row,style),add_special_tokens=False)
    if not ids or len(ids)+max_new_tokens>384:raise ValueError('Generation context budget exceeded')
    current=torch.tensor([ids],device=device);mask=torch.ones_like(current)
    past=None;generated=[];reason='length_limit';raw='';reply=''
    for _ in range(max_new_tokens):
        out=model(input_ids=current,attention_mask=mask,past_key_values=past,use_cache=True)
        logits=out.logits[:,-1,:]
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite generation logits')
        next_id=int(logits.argmax(-1));generated.append(next_id)
        past=out.past_key_values
        raw=tok.decode(generated,skip_special_tokens=True);reply=raw
        if next_id==tok.eos_token_id:reason='eos';break
        found=BOUNDARY.search(raw)
        if found:
            reply=raw[:found.start()];reason='turn_boundary';break
        current=torch.tensor([[next_id]],device=device)
        mask=torch.cat([mask,torch.ones((1,1),device=device,dtype=mask.dtype)],dim=1)
    return dict(reply=reply.strip(),raw_text=raw,termination=reason,generated_tokens=len(generated),status='failed_empty' if not reply.strip() else 'length_limit' if reason=='length_limit' else 'complete')
