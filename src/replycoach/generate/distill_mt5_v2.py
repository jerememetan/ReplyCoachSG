"""Auditable sequence-distillation targets, kept separate from source labels."""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def input_only(row):
    return {key: row[key] for key in ('item_id', 'context', 'speakers', 'message')}


def reviewed(raw, decisions, stage):
    """Require explicit, source-bound decisions; missing decisions never pass."""
    by_id = {row['item_id']: row for row in raw}
    if len(by_id) != len(raw):
        raise ValueError('Duplicate raw identifiers')
    result = {}
    seen = set()
    for decision in decisions:
        key = decision['item_id']
        if key in seen or key not in by_id:
            raise ValueError('Duplicate or unknown review identifier: ' + key)
        seen.add(key)
        row = by_id[key]
        if decision['raw_sha256'] != digest(row):
            raise ValueError('Review refers to different raw output: ' + key)
        if decision['decision'] not in ('accept', 'reject') or not decision['reason'].strip():
            raise ValueError('Explicit decision and reason required: ' + key)
        if decision['decision'] == 'reject':
            continue
        output = row[stage]
        target = decision.get('edited_target', output['reply']).strip()
        if not target or (output['status'] != 'complete' and 'edited_target' not in decision):
            raise ValueError('Incomplete generation cannot pass without an explicit correction: ' + key)
        result[key] = dict(target=target, edited=target != output['reply'].strip(), review=decision)
    return result


def student_rows(inputs, english, singlish):
    """The rewrite anchor is the accepted teacher English, never another label."""
    return [dict(**input_only(row), standard=english[row['item_id']]['target'],
                 local=singlish[row['item_id']]['target']) for row in inputs
            if row['item_id'] in english and row['item_id'] in singlish]


def common_subset(original, teachers):
    original_ids = {row['item_id'] for row in original}
    if len(original_ids) != len(original):
        raise ValueError('Duplicate source identifiers')
    common = set(original_ids)
    source = {r['item_id']: input_only(r) for r in original}
    for rows in teachers.values():
        if len({r['item_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate teacher identifiers')
        for row in rows:
            if row['item_id'] not in source or source[row['item_id']] != input_only(row):
                raise ValueError('Teacher changed conversation input')
        common &= {r['item_id'] for r in rows}
    order = [r['item_id'] for r in original if r['item_id'] in common]
    arms = dict(original=original, **teachers)
    result = {}
    for arm, rows in arms.items():
        lookup = {r['item_id']: r for r in rows}
        result[arm] = [lookup[key] for key in order]
    return result
