"""Condition Singlish generation on the standard reply the student actually chose."""
from replycoach.generate.style_control import REWRITE_STYLE


def rewrite_standard(model, tokenizer, rows, standard, max_source, max_target, device,
                     decoding=None, speaker_labels=False):
    """Rewrite supplied STANDARD drafts; never replace them with references or regenerate them."""
    from replycoach.generate.train_v5 import generate_replies
    if len(rows) != len(standard):
        raise ValueError('One STANDARD anchor per conversation required')
    if any(not isinstance(text, str) or not text.strip() for text in standard):
        raise ValueError('Nonempty STANDARD anchors required')
    anchored = [{**row, 'standard_reply': text} for row, text in zip(rows, standard)]
    return generate_replies(model, tokenizer, anchored, REWRITE_STYLE, max_source,
                            max_target, device, decoding, speaker_labels)


def emphasize_standard(rows, repeats=1):
    """Repeat STANDARD supervision while retaining each LOCAL rewrite once."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('STANDARD repeats must be a positive integer')
    return [row for row in rows for _ in range(repeats if row['style'] == 'STANDARD' else 1)]


def rewrite_rows(rows):
    pairs={}
    for row in rows:
        pair=pairs.setdefault(row['item_id'],{})
        if row['style'] in pair:raise ValueError('Rewrite training requires one pair per input')
        pair[row['style']]=row
    output=[]
    for pair in pairs.values():
        if set(pair)!={'STANDARD','LOCAL'}:raise ValueError('Complete STANDARD/LOCAL pair required')
        standard,local=pair['STANDARD'],pair['LOCAL']
        if any(standard.get(k)!=local.get(k) for k in ('context','message','speakers')):
            raise ValueError('Paired rewrite inputs must have identical context and roles')
        output += [standard,{**local,'style':REWRITE_STYLE,'standard_reply':standard['response']}]
    return output


def generate_pair(model,tokenizer,rows,max_source,max_target,device,decoding=None,speaker_labels=False):
    from replycoach.generate.train_v5 import generate_replies
    standard=generate_replies(model,tokenizer,rows,'STANDARD',max_source,max_target,device,decoding,speaker_labels)
    # No authored reference is used as the anchor at inference time.
    valid=[i for i,text in enumerate(standard) if text.strip()]
    local=['']*len(rows)
    if valid:
        rewritten=rewrite_standard(model,tokenizer,[rows[i] for i in valid],
                                   [standard[i].strip() for i in valid],max_source,max_target,
                                   device,decoding,speaker_labels)
        for i,text in zip(valid,rewritten):local[i]=text
    return {'STANDARD':standard,'LOCAL':local}
