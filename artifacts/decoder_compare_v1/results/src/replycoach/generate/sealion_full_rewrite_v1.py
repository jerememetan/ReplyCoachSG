"""Freeze full-data SEA English outputs for a separately trained LOCAL renderer."""
from copy import deepcopy


def frozen_inputs(previous_inputs, full_report):
    if full_report.get('status') != 'complete' or full_report.get('flagged_outputs'):
        raise ValueError('A complete, unflagged full-data English baseline is required')
    rows = [r for cases in full_report['outputs'].values() for r in cases]
    lookup = {r['item_id']: r for r in rows}
    if len(lookup) != len(rows) or len(rows) != 82:
        raise ValueError('Expected82 unique baseline conversations')
    if len(previous_inputs) != 82 or {r['item_id'] for r in previous_inputs} != set(lookup):
        raise ValueError('Baseline/evaluation ID coverage differs')
    result = []
    for original in previous_inputs:
        row = deepcopy(original)
        direct = lookup[row['item_id']]['styles']
        for style in ['STANDARD', 'LOCAL']:
            if direct[style]['status'] != 'complete' or not direct[style]['reply'].strip():
                raise ValueError('Incomplete direct baseline')
        row['actual_anchor'] = direct['STANDARD']['reply']
        row['direct_local'] = direct['LOCAL']['reply']
        result.append(row)
    return result
