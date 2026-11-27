"""Collect paired raw generations and a blinded review packet; do not invent scores."""
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    out = ROOT / 'review'
    out.mkdir(exist_ok=True)
    summary = dict(status='needs_attention', quality_scored=False, default_changed=False, stages={}, files={})
    for arm in ('sealion', 'qwen'):
        for stage in ('task', 'rewrite'):
            path = ROOT / 'outputs' / arm / stage / 'training.json'
            report = read(path) if path.exists() else {}
            summary['stages'][arm + '_' + stage] = {key: report.get(key) for key in
                ('status', 'selected_epoch', 'optimizer_updates', 'trainable_parameters', 'frozen_base_parameters_unchanged')}
    if not all(x['status'] == 'complete' for x in summary['stages'].values()):
        save(ROOT / 'summary.json', summary)
        return
    packet, key = [], []
    rng = random.Random(425)
    for checkpoint in ('selected', 'epoch_04'):
        data = {arm: read(ROOT / 'outputs' / arm / ('pipeline_' + checkpoint + '.json')) for arm in ('sealion', 'qwen')}
        maps = {arm: {r['item_id']: r for r in result['outputs']} for arm, result in data.items()}
        if len(maps['sealion']) != 75 or maps['sealion'].keys() != maps['qwen'].keys():
            raise ValueError('Missing or mismatched comparison outputs')
        rows = []
        for ident, sea in maps['sealion'].items():
            qwen = maps['qwen'][ident]
            if sea['context'] != qwen['context'] or sea['message'] != qwen['message']:
                raise ValueError('Input mismatch')
            rows.append(dict(item_id=ident, context=json.dumps(sea['context'], ensure_ascii=False), message=sea['message'],
                             sealion_english=sea['english']['reply'], sealion_singlish=sea['singlish']['reply'],
                             qwen_english=qwen['english']['reply'], qwen_singlish=qwen['singlish']['reply']))
            arms = ['sealion', 'qwen']
            rng.shuffle(arms)
            blind_id = hashlib.sha256((checkpoint + ident).encode()).hexdigest()[:16]
            packet.append(dict(id=blind_id, checkpoint_policy=checkpoint, context=sea['context'], speakers=sea['speakers'],
                               message=sea['message'], candidates={label: dict(english=maps[arm][ident]['english']['reply'],
                               singlish=maps[arm][ident]['singlish']['reply']) for label, arm in zip(('A', 'B'), arms)}))
            key.append(dict(id=blind_id, item_id=ident, A=arms[0], B=arms[1]))
        with (out / ('comparison_' + checkpoint + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary[checkpoint + '_flags'] = {arm: value['flags'] for arm, value in data.items()}
    rng.shuffle(packet)
    save(out / 'blinded_packet.json', packet)
    save(out / 'PRIVATE_key.json', key)
    summary.update(status='complete_unscored', scope='Reused development cases; raw generations and review packet, no independent human ratings')
    save(ROOT / 'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
