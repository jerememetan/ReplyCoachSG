"""Build matched student arms only from fully reviewed teacher generations."""
import argparse
import json
from pathlib import Path

from replycoach.generate.distill_mt5_v2 import common_subset, reviewed, student_rows
from replycoach.generate.decoder_compare_v1 import write_json
from run_decoder_compare_v1 import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['pilot', 'full'], required=True)
    args = parser.parse_args()
    recipe = json.loads((ROOT / 'recipe.json').read_text())
    inputs = json.loads((ROOT / 'data/train_inputs.json').read_text())
    if args.scope == 'pilot':
        inputs = [r for r in inputs if r['item_id'] in recipe['pilot_ids']]
    wanted = {r['item_id'] for r in inputs}
    evidence = {ROOT / 'recipe.json', ROOT / 'data/train_inputs.json', ROOT / 'data/train_rewrite.json'}
    original = [r for r in json.loads((ROOT / 'data/train_rewrite.json').read_text()) if r['item_id'] in wanted]
    teachers = {}
    stats = {}
    for arm in ('sealion', 'qwen'):
        approved = {}
        for stage in ('english', 'singlish'):
            rawpath = ROOT / f'outputs/{arm}/{stage}_{args.scope}.json'
            reviewpath = ROOT / f'reviews/{arm}/{stage}_{args.scope}.json'
            report = json.loads(rawpath.read_text())
            decisions = json.loads(reviewpath.read_text())['decisions']
            evidence.update([rawpath, reviewpath])
            raw = report['outputs']
            raw_ids = {r['item_id'] for r in raw}
            expected = wanted if stage == 'english' else set(approved['english'])
            if report['status'] != 'complete' or raw_ids != expected:
                raise ValueError('Missing/extra teacher generation: ' + arm + '/' + stage)
            if len(decisions) != len(raw) or {r['item_id'] for r in decisions} != raw_ids:
                raise ValueError('Every raw output requires a review decision')
            if stage == 'singlish':
                for row in raw:
                    if row['standard_anchor'] != approved['english'][row['item_id']]['target']:
                        raise ValueError('Singlish generated from the wrong accepted English')
            approved[stage] = reviewed(raw, decisions, stage)
            stats[arm + '_' + stage] = dict(generated=len(raw), accepted=len(approved[stage]),
                                            rejected=len(raw) - len(approved[stage]),
                                            edited=sum(r['edited'] for r in approved[stage].values()))
        teachers[arm] = student_rows(inputs, approved['english'], approved['singlish'])
    arms = common_subset(original, teachers)
    destination = ROOT / f'targets/{args.scope}'
    if destination.exists():
        raise FileExistsError('Target snapshot already exists')
    for arm, rows in arms.items():
        if not rows:
            raise ValueError('Empty common student subset')
        path = destination / (arm + '.json')
        write_json(path, rows)
        evidence.add(path)
    common = [r['item_id'] for r in arms['original']]
    manifest = dict(scope=args.scope, source_conversations=len(inputs), original_eligible=len(original),
                    common_conversations=len(common), common_ids=common, review_counts=stats,
                    excluded_ids=[r['item_id'] for r in inputs if r['item_id'] not in set(common)],
                    original_provenance='Previously audited766 paired labels, mixed human/synthetic/assistant-revised; no new target edits.',
                    independent_human_gold=False,
                    files={str(p.relative_to(ROOT)): sha(p) for p in sorted(evidence)})
    write_json(destination / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
