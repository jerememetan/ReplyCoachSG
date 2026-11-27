"""Load audited distillation targets anchored to exactly the original train inputs."""
from hashlib import sha256
import json
from pathlib import Path


def load_teacher_targets(path: Path, train_rows: list[dict], train_path: Path):
    report=json.loads(path.read_text())
    if report.get('schema_version')!=1 or report.get('status')!='audited':
        raise ValueError('Teacher targets must be explicitly audited')
    if report.get('source_train_sha256')!=sha256(train_path.read_bytes()).hexdigest():
        raise ValueError('Teacher targets belong to a different training source')
    if not report.get('teacher_settings',{}).get('model_digest'):
        raise ValueError('Teacher model digest is required')
    expected={}
    for r in train_rows:
        expected.setdefault(r['item_id'],{})[r['style']]=r
    items=report.get('items')
    if not isinstance(items,list) or len(items)!=len(expected):
        raise ValueError('Exactly one teacher pair per original training input required')
    rows=[];seen=set()
    for c in items:
        item_id=c.get('item_id')
        if item_id not in expected or item_id in seen:raise ValueError('Unknown or duplicate teacher item ID')
        seen.add(item_id)
        reference=expected[item_id]
        if set(reference)!={'STANDARD','LOCAL'}:raise ValueError('Original paired styles required')
        if any(c.get(k)!=reference['STANDARD'][k] for k in ('context','speakers','message')):
            raise ValueError('Teacher may change targets only, never original inputs or roles')
        review=c.get('review',{})
        if review.get('approved') is not True or not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip():
            raise ValueError('Every teacher item requires explicit audit approval')
        for k in ('meaning_preserved','speaker_correct','natural_local','context_compatible'):
            if review.get(k) is not True:raise ValueError('Teacher audit checks must all be true')
        replies=c.get('replies',{})
        if set(replies)!={'standard','local'} or any(not isinstance(v,str) or not v.strip() for v in replies.values()):
            raise ValueError('Two nonempty teacher replies required')
        if replies['standard'].strip().casefold()==replies['local'].strip().casefold():
            raise ValueError('Teacher style wording must differ')
        rows.extend({**reference[s],'response':replies[s.lower()]} for s in ('STANDARD','LOCAL'))
    return rows,{'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest(),'items':len(items),
                 'teacher_model_digest':report['teacher_settings']['model_digest']}
