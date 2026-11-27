"""Deterministic beam decoding for SEA-LION's original tuple KV cache.

Keeps the existing EOS/transcript-boundary stop rules. Scores are cumulative
FP32 log probability divided by generated-token count**length_penalty. Finished
beams are retained separately; active beams use an admissible max-length bound.
"""
import math
import torch
from torch.nn import functional as F


def reorder_legacy_cache(cache, parents):
    return [tuple(t.index_select(0, parents) for t in layer) for layer in cache]


def banned_next(tokens, size):
    if size <= 0 or len(tokens) < size - 1:
        return set()
    prefix = tokens[-(size-1):] if size > 1 else []
    return {tokens[i+size-1] for i in range(len(tokens)-size+1)
            if tokens[i:i+size-1] == prefix}


def rendered(tok, tokens, boundary):
    raw = tok.decode(tokens, skip_special_tokens=True)
    found = boundary.search(raw)
    if tokens and tokens[-1] == tok.eos_token_id:
        return raw, raw, 'eos'
    if found:
        return raw, raw[:found.start()], 'turn_boundary'
    return raw, raw, None


@torch.no_grad()
def beam_decode(model, tok, prompt_ids, boundary, num_beams=4, max_new_tokens=64,
                length_penalty=1.0, no_repeat_ngram_size=0, device='cuda'):
    if num_beams < 2 or max_new_tokens < 1 or length_penalty < 0:
        raise ValueError('Use at least2 beams, positive budget and nonnegative length penalty')
    if not prompt_ids or len(prompt_ids)+max_new_tokens > 384:
        raise ValueError('Generation context budget exceeded')
    model.eval();current=torch.tensor([prompt_ids],device=device)
    past=None;active=[([],0.)];finished=[];forward_calls=0
    def score(candidate):return candidate[1]/max(1,len(candidate[0]))**length_penalty
    for position in range(max_new_tokens):
        mask=torch.ones((len(active),len(prompt_ids)+position),device=device,dtype=torch.long)
        out=model(input_ids=current,attention_mask=mask,past_key_values=past,use_cache=True)
        forward_calls+=1;logits=out.logits[:,-1,:].float()
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite beam logits')
        logp=F.log_softmax(logits,dim=-1)
        for i,(tokens,_) in enumerate(active):
            blocked=banned_next(tokens,no_repeat_ngram_size)
            if blocked:logp[i,list(blocked)]=-torch.inf
        cumulative=logp+torch.tensor([x[1] for x in active],device=device,dtype=torch.float32)[:,None]
        width=logp.shape[-1];k=min(2*num_beams,cumulative.numel())
        values,indices=torch.topk(cumulative.reshape(-1),k)
        next_active=[];parents=[]
        for value,index in zip(values.tolist(),indices.tolist(),strict=True):
            if not math.isfinite(value):continue
            parent,token=divmod(index,width);tokens=active[parent][0]+[token]
            _,_,reason=rendered(tok,tokens,boundary)
            if reason:finished.append((tokens,value,reason))
            elif position+1==max_new_tokens:finished.append((tokens,value,'length_limit'))
            elif len(next_active)<num_beams:
                next_active.append((tokens,value));parents.append(parent)
        finished=sorted(finished,key=score,reverse=True)[:num_beams]
        if not next_active:break
        if finished and score(finished[0]) >= max(v/max_new_tokens**length_penalty for _,v in next_active):
            break
        parent_ids=torch.tensor(parents,device=device,dtype=torch.long)
        past=reorder_legacy_cache(out.past_key_values,parent_ids)
        active=next_active
        current=torch.tensor([[tokens[-1]] for tokens,_ in active],device=device)
    if not finished:raise RuntimeError('No finished or capped beam candidates')
    tokens,value,reason=max(finished,key=score);raw,reply,_=rendered(tok,tokens,boundary)
    return dict(reply=reply.strip(),raw_text=raw,termination=reason,
                generated_tokens=len(tokens),token_ids=tokens,
                cumulative_log_probability=value,normalized_score=score((tokens,value)),
                forward_calls=forward_calls,status='failed_empty' if not reply.strip() else
                'length_limit' if reason=='length_limit' else 'complete')
