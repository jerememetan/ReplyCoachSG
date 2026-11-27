"""Shared data and decoding rules for the SEA-LION/Qwen base-model comparison."""
import math

from replycoach.generate.sealion_audited_restart_v1 import conversation, prompt


def encode(row, tokenizer, stage, style=None, limit=384):
    if stage == 'rewrite':
        text = prompt(row, 'LOCAL_REWRITE', row['standard'])
        target = row['local']
    elif stage == 'task' and style in ('STANDARD', 'LOCAL'):
        text = prompt(row, style)
        target = row[style.lower()]
    else:
        raise ValueError('Unknown training route')
    source = tokenizer.encode(text, add_special_tokens=False)
    reply = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not source or not target.strip() or len(source) + len(reply) > limit:
        raise ValueError('Empty or over-budget example: ' + row['item_id'])
    return dict(input_ids=source + reply, attention_mask=[1] * (len(source) + len(reply)),
                labels=[-100] * len(source) + reply, prompt_tokens=len(source))


def lr_multiplier(update, total, warmup):
    if not 1 <= update <= total or not 1 <= warmup < total:
        raise ValueError('Invalid optimizer-update schedule')
    if update <= warmup:
        return update / warmup
    return .5 * (1 + math.cos(math.pi * (update - warmup) / (total - warmup)))


def validate_groups(train, rewrite, old, development, evaluation, references):
    """Evaluation deliberately contains old validation: disclose, do not relabel it test."""
    def unique(rows):
        ids = [r['item_id'] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate conversation ID')
        for r in rows:
            conversation(r)
        return set(ids)
    groups = [unique(rows) for rows in (train, old, development, evaluation)]
    if any(groups[0] & other for other in groups[1:]):
        raise ValueError('Training/evaluation ID leakage')
    training_text = {conversation(r) for r in train}
    if any(training_text & {conversation(r) for r in rows}
           for rows in (old, development, evaluation)):
        raise ValueError('Training/evaluation conversation leakage')
    train_by_id = {r['item_id']: r for r in train}
    if not unique(rewrite) <= groups[0] or any(r != train_by_id[r['item_id']] for r in rewrite):
        raise ValueError('Rewrite training does not match task training')
    if not unique(references) <= groups[3]:
        raise ValueError('Reference diagnostics must be a subset of evaluation')
    if groups[1] & groups[2]:
        raise ValueError('Validation groups overlap')


def reorder_cache(cache, parents):
    if hasattr(cache, 'reorder_cache'):
        cache.reorder_cache(parents)
        return cache
    return [tuple(t.index_select(0, parents) for t in layer) for layer in cache]


def write_json(path, value):
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)
