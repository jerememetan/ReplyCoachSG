"""Seeded nucleus sampling with the original SEA-LION cache/stop rules."""


def nucleus_probabilities(logits,temperature=.8,top_p=.9):
    import torch
    if temperature<=0 or not 0<top_p<=1:raise ValueError('Invalid sampling settings')
    ordered,ids=torch.sort(logits.float()/temperature,descending=True)
    probs=ordered.softmax(-1)
    remove=probs.cumsum(-1)>top_p
    remove[1:]=remove[:-1].clone();remove[0]=False
    probs=probs.masked_fill(remove,0);probs=probs/probs.sum()
    return ids,probs


def sample_decode(model,tok,prompt_ids,boundary,seed,temperature=.8,top_p=.9,
                  max_new_tokens=64,no_repeat_ngram_size=3,device='cuda'):
    import torch
    from replycoach.generate.sealion_beam_v1 import banned_next,rendered
    if not prompt_ids or len(prompt_ids)+max_new_tokens>384:raise ValueError('Context budget exceeded')
    generator=torch.Generator(device=device).manual_seed(seed)
    current=torch.tensor([prompt_ids],device=device);past=None;generated=[];reason='length_limit'
    model.eval()
    with torch.no_grad():
        for i in range(max_new_tokens):
            mask=torch.ones((1,len(prompt_ids)+i),device=device,dtype=torch.long)
            out=model(input_ids=current,attention_mask=mask,past_key_values=past,use_cache=True)
            logits=out.logits[0,-1,:].float()
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite logits')
            blocked=banned_next(generated,no_repeat_ngram_size)
            if blocked:logits[list(blocked)]=-torch.inf
            ids,probs=nucleus_probabilities(logits,temperature,top_p)
            token=int(ids[torch.multinomial(probs,1,generator=generator)])
            generated.append(token);raw,reply,stopped=rendered(tok,generated,boundary)
            if stopped:reason=stopped;break
            past=out.past_key_values;current=torch.tensor([[token]],device=device)
    return dict(reply=reply.strip(),raw_text=raw,token_ids=generated,seed=seed,
        termination=reason,status='failed_empty' if not reply.strip() else 'length_limit' if reason=='length_limit' else 'complete',
        temperature=temperature,top_p=top_p)
