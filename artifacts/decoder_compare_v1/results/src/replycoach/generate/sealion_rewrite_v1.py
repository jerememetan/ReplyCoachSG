"""English-anchored SEA-LION LOCAL rendering, with prompt-only masking."""
import re
import torch
from replycoach.generate.sealion_data import source
BOUNDARY=re.compile(r'\n(?:Friend|Other|English draft|You(?: \([A-Z_]+\))?):')

def rewrite_source(row,anchor):
    if not isinstance(anchor,str) or not anchor.strip():raise ValueError('Nonempty English anchor required')
    conversation=source(row,'STANDARD').rsplit('\n',1)[0]
    return ('Rewrite the English draft in natural Singlish. Preserve its people, facts, timing and decision.\n'
        +conversation+'\nEnglish draft: '+anchor.strip()+'\nYou (LOCAL_REWRITE):')

def encode_rewrite(row,tok,limit=384):
    prompt=tok.encode(rewrite_source(row,row['standard_anchor']),add_special_tokens=False)
    if not row['response'].strip():raise ValueError('Nonempty LOCAL target required')
    reply=tok.encode(row['response'],add_special_tokens=False)+[tok.eos_token_id]
    if len(prompt)+len(reply)>limit:raise ValueError('No truncation: '+row['item_id'])
    return dict(input_ids=prompt+reply,attention_mask=[1]*(len(prompt)+len(reply)),labels=[-100]*len(prompt)+reply,prompt_tokens=len(prompt))

@torch.no_grad()
def generate_rewrite(model,tok,row,anchor,max_new_tokens=64,device='cuda'):
    model.eval();ids=tok.encode(rewrite_source(row,anchor),add_special_tokens=False)
    if not ids or len(ids)+max_new_tokens>384:raise ValueError('Generation context budget exceeded')
    current=torch.tensor([ids],device=device);mask=torch.ones_like(current);past=None;generated=[];reason='length_limit';raw='';reply=''
    for _ in range(max_new_tokens):
        out=model(input_ids=current,attention_mask=mask,past_key_values=past,use_cache=True);logits=out.logits[:,-1,:]
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite logits')
        next_id=int(logits.argmax(-1));generated.append(next_id);past=out.past_key_values
        raw=tok.decode(generated,skip_special_tokens=True);reply=raw
        if next_id==tok.eos_token_id:reason='eos';break
        found=BOUNDARY.search(raw)
        if found:reply=raw[:found.start()];reason='turn_boundary';break
        current=torch.tensor([[next_id]],device=device);mask=torch.cat([mask,torch.ones((1,1),device=device,dtype=mask.dtype)],dim=1)
    return dict(reply=reply.strip(),raw_text=raw,termination=reason,generated_tokens=len(generated),status='failed_empty' if not reply.strip() else 'length_limit' if reason=='length_limit' else 'complete',standard_anchor=anchor)
