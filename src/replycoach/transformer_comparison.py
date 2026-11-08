"""Blinded validation comparison; personas are nested within conversations."""
from collections import defaultdict
import json
import random
from statistics import mean

METRICS = ('appropriateness', 'naturalness', 'style_fit', 'would_send')


def task_key(candidate, persona_id):
    # Match judge_one's actual task serialization, excluding anonymous identifiers.
    task = {k: candidate[k] for k in ('conversation', 'incoming_message', 'style', 'reply')}
    return persona_id, json.dumps(task, ensure_ascii=False)


def build_packet(inputs, outputs, seed=425):
    pairs = []
    for i, item in enumerate(inputs, 1):
        for model, rows in outputs.items():
            output = rows[item['item_id']]
            for style in ['STANDARD', 'LOCAL']:
                public = dict(schema_version=1, conversation_id=f'Q{i:06d}',
                              conversation=item['conversation'],
                              incoming_message={'speaker': 'peer', 'text': item['message']},
                              style=style, reply=output[style.lower()])
                private = dict(schema_version=1, conversation_id=public['conversation_id'],
                               item_id=item['item_id'], style=style, system=model,
                               memory_mode='M2', retrieval_method='none', requested_mode='generator_only',
                               evaluation_split='validation', generation_error=output.get('error'))
                pairs.append((public, private))
    random.Random(seed).shuffle(pairs)
    for i, (public, private) in enumerate(pairs, 1):
        public['candidate_id'] = private['candidate_id'] = f'C{i:06d}'
    return [p for p, _ in pairs], [p for _, p in pairs]


def summarise_comparison(public, private, ratings, outputs, personas, bootstrap_samples=5000):
    mapping = {r['candidate_id']: r for r in private}
    latest = {(r['candidate_id'], r['persona_id']): r for r in ratings}
    expected = {(r['candidate_id'], p) for r in public for p in personas}
    if set(latest) != expected or any(r['status'] != 'ok' for r in latest.values()):
        raise ValueError('Need complete matched ratings before recommending a model')
    cells = {}
    for candidate in public:
        meta = mapping[candidate['candidate_id']]
        values = [latest[(candidate['candidate_id'], p)]['rating'] for p in personas]
        cells[(meta['system'], meta['item_id'], meta['style'])] = {
            metric: mean(v[metric] for v in values) for metric in METRICS}
    items = sorted({r['item_id'] for r in private})
    summary, overall = [], {}
    for model in outputs:
        by_style = {}
        for style in ['STANDARD', 'LOCAL']:
            rows = [cells[model, item, style] for item in items]
            by_style[style] = {m: mean(r[m] for r in rows) for m in METRICS}
            summary.append(dict(model=model, style=style, conversations=len(items), **by_style[style]))
        overall[model] = {m: mean(s[m] for s in by_style.values()) for m in METRICS}
    ranked = sorted(overall, key=lambda m: (-overall[m]['appropriateness'], m))
    top = ranked[0]
    paired = []
    for other in ranked[1:]:
        differences = [mean(cells[top, item, style]['appropriateness'] -
                            cells[other, item, style]['appropriateness']
                            for style in ['STANDARD', 'LOCAL']) for item in items]
        rng = random.Random(425)
        boot = sorted(mean(rng.choices(differences, k=len(items))) for _ in range(bootstrap_samples))
        interval = [boot[int(.025 * len(boot))], boot[min(len(boot)-1, int(.975 * len(boot)))]]
        paired.append(dict(top_model=top, other_model=other, appropriateness_delta=mean(differences),
                           appropriateness_delta_ci95=interval))
    return dict(completed=True, evaluation_split='validation', conversations=len(items),
                expected_ratings=len(expected), persona_count=len(personas), summary=summary,
                overall=overall, highest_point_estimate=top,
                recommended_model=top if all(r['appropriateness_delta_ci95'][0] > 0 for r in paired) else None,
                primary_criterion='Mean appropriateness, equally weighted Standard/Local; paired conversation bootstrap',
                top_model_paired_comparisons=paired,
                generation_failures={m: sum(bool(r.get('error')) for r in rows.values()) for m, rows in outputs.items()},
                limitation='Correlated fictional AI personas, not independent human evidence. Validation model selection, not a held-out test.')
