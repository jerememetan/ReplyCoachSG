"""Reference-free conversation inputs and reply-only causal training labels."""
SPEAKERS={'user':'You','peer':'Friend','other':'Other'}


def source(row,style='LOCAL'):
    # DIALOGUE is a training-only public adaptation tag. Final task inference
    # still permits STANDARD/LOCAL only in sealion_dual_v1.generate_reply.
    if style not in {'LOCAL','STANDARD','DIALOGUE'}:raise ValueError(style)
    if len(row['context'])!=len(row['speakers']):raise ValueError('Speaker coverage mismatch')
    turns=[f'{SPEAKERS[s]}: {text}' for s,text in zip(row['speakers'],row['context'],strict=True)]
    turns.append('Friend: '+row['message'])
    return '\n'.join([*turns,f'You ({style}):'])


def encode_reply(row,tok,limit=384):
    prompt=tok.encode(source(row,style=row.get('style','LOCAL')),add_special_tokens=False)
    reply=tok.encode(row['response'],add_special_tokens=False)+[tok.eos_token_id]
    if not prompt or not row['response'].strip():raise ValueError('Nonempty prompt and reply required')
    ids=prompt+reply
    if len(ids)>limit:raise ValueError('Source/target would be truncated: '+row['item_id'])
    return dict(input_ids=ids,attention_mask=[1]*len(ids),labels=[-100]*len(prompt)+reply,prompt_tokens=len(prompt))


def pad_example(example,pad_id,limit=384):
    size=len(example['input_ids'])
    if size>limit:raise ValueError('Padding limit exceeded')
    extra=limit-size
    return dict(input_ids=example['input_ids']+[pad_id]*extra,
        attention_mask=example['attention_mask']+[0]*extra,
        labels=example['labels']+[-100]*extra)
