"""Match reviewed teachers against the original paired-label eligibility set."""
from replycoach.generate.distill_mt5_v2 import common_subset, input_only


def match_reviewed_arms(original, teachers, inputs):
    """Validate the full input pool before excluding original-ineligible rows.

    Teachers generate from 768 inputs, while the original arm has only 766
    eligible pairs. An ineligible pair is not an unknown conversation, but its
    identity must still be checked before it is excluded from every arm.
    """
    source = {row['item_id']: input_only(row) for row in inputs}
    if len(source) != len(inputs):
        raise ValueError('Duplicate source input identifiers')
    eligible = {row['item_id'] for row in original}
    for arm, rows in dict(original=original, **teachers).items():
        if len({row['item_id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate identifiers in ' + arm)
        for row in rows:
            if source.get(row['item_id']) != input_only(row):
                raise ValueError('Unknown or changed conversation input: ' + arm + '/' + row['item_id'])
    eligible_teachers = {
        arm: [row for row in rows if row['item_id'] in eligible]
        for arm, rows in teachers.items()
    }
    return common_subset(original, eligible_teachers)
