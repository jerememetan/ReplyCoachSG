"""Descriptive, read-only audit of the actual project targets and saved replies.

Surface features are diagnostics, never human Singlish-quality judgments.
"""
from collections import Counter
from pathlib import Path
from statistics import median
import csv
import hashlib
import json
import random
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/report_section_3_3_20261121/distillation_update'
MARKERS = re.compile(r'\b(?:lah|leh|lor|sia|liao|paiseh|makan|dabao|tabao|jio|anot|chope|bochap|sian)\b|can or not', re.I)
CASUAL = re.compile(r'\b(?:alr|tmr|cmi|rmb|pls|plz|u|ur|rn|cos|abt|dun|nvr|gonna|wanna)\b', re.I)


def read(path):
    path = ROOT / path
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.suffix == '.jsonl' else json.loads(path.read_text())


def normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def pair(row):
    if 'replies' in row:
        return row['replies']['standard'], row['replies']['local']
    return row['standard'], row['local']


def features(standard, local):
    return {
        'exact_copy': standard == local,
        'case_punctuation_normalized_copy': normalized(standard) == normalized(local),
        'narrow_local_marker': bool(MARKERS.search(local)),
        'casual_abbreviation': bool(CASUAL.search(local)),
        'starts_can': bool(re.match(r'^can\b', local, re.I)),
        'contains_i_will': bool(re.search(r'\bi will\b', local, re.I)),
        'words': len(local.split()),
    }


def main():
    OUT.mkdir(exist_ok=True)
    paths = {
        'original190': 'data/dev/train.jsonl',
        'synthetic600_before_style_audit': 'data/training_expansion/generated600_repair_v1/generated_corrected.jsonl',
        'actual_larger_model_task768': 'artifacts/qwen_full_sealion_recipe_v1/results/data/train_task.json',
        'actual_larger_model_rewrite766': 'artifacts/qwen_full_sealion_recipe_v1/results/data/train_rewrite.json',
        'distillation_sealion571': 'artifacts/distill_mt5_v2/results/targets/full/sealion.json',
        'distillation_qwen571': 'artifacts/distill_mt5_v2/results/targets/full/qwen.json',
        'validation50': 'data/dev/validation.jsonl',
    }
    datasets = {name: read(path) for name, path in paths.items()}
    actual = datasets['actual_larger_model_rewrite766']
    datasets['actual_rewrite_team_original'] = [r for r in actual if r['item_id'].startswith('ITEM-')]
    datasets['actual_rewrite_synthetic'] = [r for r in actual if r['item_id'].startswith('SYN-')]
    population_ids = {r['id'] for r in datasets['validation50']}
    for arm in ['sealion', 'qwen']:
        path = f'artifacts/mt5_distill_judge_v1/teacher_comparison/{arm}/pipeline.json'
        paths[f'{arm}_teacher_saved75'] = path
        raw = read(path)['outputs']
        outputs = []
        for row in raw:
            if row['item_id'] not in population_ids:
                continue
            get = lambda style: row[style]['reply'] if isinstance(row[style], dict) else row[style]
            assert row['standard_anchor'] == get('english')
            outputs.append({'item_id': row['item_id'], 'standard': get('english'), 'local': get('singlish')})
        assert len(outputs) == 50
        datasets[f'{arm}_teacher_validation50_outputs'] = outputs
    all_rows = []
    summary = {}
    for name, rows in datasets.items():
        extracted = []
        for row in rows:
            standard, local = pair(row)
            record = {'dataset': name, 'item_id': row.get('item_id', row.get('id')), 'standard': standard, 'local': local, **features(standard, local)}
            extracted.append(record)
            all_rows.append(record)
        counts = {key: sum(r[key] for r in extracted) for key in features('', '') if key != 'words'}
        summary[name] = {'rows': len(rows), **counts,
                         **{key + '_pct': round(100 * value / len(rows), 2) for key, value in counts.items()},
                         'median_local_words': median(r['words'] for r in extracted),
                         'item_prefixes': dict(Counter('team' if r['item_id'].startswith('ITEM-') else 'synthetic' if r['item_id'].startswith('SYN-') else 'other' for r in extracted))}
    sample = []
    for group in ['actual_rewrite_team_original', 'actual_rewrite_synthetic']:
        for row in random.Random(425).sample(sorted(datasets[group], key=lambda r: r['item_id']), 20):
            sample.append({'group': group, **row, 'features': features(*pair(row))})
    copied_examples = [r for r in all_rows if r['dataset'] in ['actual_larger_model_rewrite766', 'sealion_teacher_validation50_outputs', 'qwen_teacher_validation50_outputs'] and r['case_punctuation_normalized_copy']]
    report = {
        'scope': 'Actual frozen target files used by the larger-model recipe, retained distillation targets, original controls and the same 50 saved teacher outputs.',
        'method': 'All-row deterministic counts; normalization lowercases and removes punctuation. Manual-review packet is 20 random original and 20 random synthetic actual rewrite rows, seed425, sampled separately.',
        'local_marker_regex': MARKERS.pattern,
        'caveats': [
            'Particles are optional; absence of these narrow markers does not establish absence of Singlish syntax.',
            'English texting abbreviations are counted separately and do not establish Singlish.',
            'An identical short reply can be natural in both registers; copying is a style-distinction diagnostic, not an error label.',
            'Incoming-message language tags describe inputs, not the strength of target-reply style.',
            'Training sources are mixed team-authored, synthetic and assistant-reviewed; not all human gold.',
            'The stratified 40-row manual sample is not a prevalence estimate.',
            'This audit does not train models, edit targets or isolate a causal effect of data on output style.'
        ],
        'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths.values()},
        'summary': summary,
    }
    (OUT / 'dataset_style_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'manual_review_packet40.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'normalized_copy_examples.json').write_text(json.dumps(copied_examples, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'dataset_style_rows.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
