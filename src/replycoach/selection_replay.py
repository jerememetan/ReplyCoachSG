"""Evaluation-only selection replay over a complete, frozen four-source packet."""
from collections import Counter, defaultdict
from time import perf_counter

import numpy as np

from replycoach.all_models import SOURCE_NAMES, align_memory_speakers
from replycoach.pipeline import prepare_input
from replycoach.schema import Candidate, STYLES
from replycoach.shared_ranker import rank_proposals

METRICS = ('appropriateness', 'naturalness', 'style_fit', 'would_send_pct')
POLICIES = ('shared', 'fixed_v5', 'fixed_minilm', 'legacy')


def validate_replay_inputs(public, mapping, ratings, persona_ids):
    pmap = {r['candidate_id']: r for r in public}
    if len(pmap) != len(public) or set(pmap) != set(mapping) or not pmap:
        raise ValueError('packet candidate coverage is incomplete or duplicated')
    if not persona_ids or len(set(persona_ids)) != len(persona_ids):
        raise ValueError('persona coverage must be nonempty and unique')
    by_cell, per_item, score_rows = {}, {}, {}
    for cid, p in pmap.items():
        meta = mapping[cid]
        model, style, item = meta['system'].removesuffix('_M2'), p['style'], meta['item_id']
        if model not in SOURCE_NAMES or style not in STYLES or meta['memory_mode'] != 'M2':
            raise ValueError('replay requires exactly BM25, MiniLM, V4 and V5 with M2')
        c = Candidate.from_dict(meta['candidate'])
        if c.candidate_text != p['reply'] or c.style != style or meta['style'] != style:
            raise ValueError('saved candidate differs from the anonymous packet')
        if not np.isfinite(c.score):
            raise ValueError('original candidate score must be finite')
        cell = item, model, style
        if cell in by_cell:
            raise ValueError('duplicate model/conversation/style candidate')
        by_cell[cell] = cid
        context = (p['conversation'], p['incoming_message'])
        if item in per_item and per_item[item] != context:
            raise ValueError('sources do not share the same conversation input')
        per_item[item] = context
    expected = {(item, model, style) for item in per_item for model in SOURCE_NAMES for style in STYLES}
    if set(by_cell) != expected:
        raise ValueError('every conversation needs all eight source proposals')
    for row in ratings:
        key = row['candidate_id'], row['persona_id']
        if key in score_rows or key[0] not in pmap or key[1] not in persona_ids or row['status'] != 'ok':
            raise ValueError('ratings contain missing, duplicate, failed or unknown pairs')
        rating = row['rating']
        if (any(type(rating[k]) is not int or not 1 <= rating[k] <= 5 for k in METRICS[:3])
                or type(rating['would_send']) is not bool):
            raise ValueError('ratings must use the existing score schema')
        score_rows[key] = [*[rating[k] for k in METRICS[:3]], 100 * rating['would_send']]
    if len(score_rows) != len(public) * len(persona_ids):
        raise ValueError('ratings do not completely cover every candidate and persona')
    return pmap, by_cell, per_item, score_rows


def replay_selection(public, mapping, ratings, persona_ids, scorer, *, bootstrap_samples=20000, seed=425):
    pmap, by_cell, per_item, score_rows = validate_replay_inputs(public, mapping, ratings, persona_ids)
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError('bootstrap sample count must be a positive integer')
    items = sorted(per_item)
    values = np.empty((len(items), len(POLICIES), 2, 4))
    selections, origin_counts, contributor_counts = [], defaultdict(Counter), defaultdict(Counter)
    started = perf_counter()
    for i, item in enumerate(items):
        history, incoming = per_item[item]
        session = [turn['text'] for turn in history]
        prepared = prepare_input(incoming['text'], session, 'M2')
        speakers = align_memory_speakers(session, [turn['speaker'] for turn in history], prepared.memory)
        proposals = {model: [Candidate.from_dict(mapping[by_cell[item, model, style]]['candidate'])
                             for style in STYLES] for model in SOURCE_NAMES}
        if any(c.context != prepared.memory for pair in proposals.values() for c in pair):
            raise ValueError('saved proposals and current M2 memory differ')
        ranking = rank_proposals(incoming['text'], prepared.memory, proposals, scorer, speakers)
        for t, style in enumerate(STYLES):
            chosen = getattr(ranking, style.lower())
            matching = [by_cell[item, model, style] for model in chosen.meta['contributing_models']]
            # Anonymous packet order, never a rating, breaks duplicate-record ties.
            shared_id = min(matching)
            legacy_id = min((by_cell[item, model, style] for model in SOURCE_NAMES),
                            key=lambda cid: (-mapping[cid]['candidate']['score'], pmap[cid]['reply'], cid))
            ids = {'shared': shared_id, 'fixed_v5': by_cell[item, 'V5', style],
                   'fixed_minilm': by_cell[item, 'MiniLM', style], 'legacy': legacy_id}
            for p, policy in enumerate(POLICIES):
                values[i, p, t] = np.mean([score_rows[ids[policy], persona] for persona in persona_ids], axis=0)
            meta = mapping[shared_id]
            origin = meta['candidate'].get('meta', {}).get('candidate_source', meta['candidate']['source'])
            origin_counts[style][origin] += 1
            contributor_counts[style].update(chosen.meta['contributing_models'])
            selections.append({'item_id': item, 'style': style, 'shared_candidate_id': shared_id,
                               'shared_matching_ids': sorted(matching), 'shared_models': chosen.meta['contributing_models'],
                               'shared_origin': origin, 'shared_reply': chosen.candidate_text,
                               'shared_score': chosen.score, 'truncated': chosen.meta['truncated'],
                               'policy_candidate_ids': ids})
    seconds = perf_counter() - started
    summary = [{'policy': policy, 'style': style, 'conversations': len(items),
                **dict(zip(METRICS, map(float, values[:, p, t].mean(axis=0))))}
               for p, policy in enumerate(POLICIES) for t, style in enumerate(STYLES)]
    weights = np.random.default_rng(seed).multinomial(len(items), np.full(len(items), 1/len(items)),
                                                      size=bootstrap_samples) / len(items)
    comparison = {}
    for p, policy in enumerate(POLICIES[1:], 1):
        comparison['shared_minus_' + policy] = {}
        for t, style in enumerate(STYLES):
            differences = values[:, 0, t] - values[:, p, t]
            interval = np.percentile(weights @ differences, [2.5, 97.5], axis=0)
            comparison['shared_minus_' + policy][style] = {
                metric: {'difference': float(differences[:, m].mean()),
                         'ci95': [float(x) for x in interval[:, m]]}
                for m, metric in enumerate(METRICS)}
    return {'record_type': 'all_models_selection_replay', 'ranker': dict(scorer.identity),
            'conversations': len(items), 'proposals': len(public), 'existing_ratings': len(ratings),
            'new_judge_calls': 0, 'ranking_seconds': seconds, 'bootstrap_samples': bootstrap_samples,
            'bootstrap_seed': seed, 'summary': summary, 'comparison': comparison, 'selections': selections,
            'selected_origin_counts': {style: dict(c) for style, c in origin_counts.items()},
            'contributor_counts': {style: dict(c) for style, c in contributor_counts.items()},
            'limitations': ['Counterfactual selection on frozen saved proposals, not a new live-model evaluation.',
                            'AI ratings remain evaluation records and are never supplied to the scorer.',
                            'Duplicate text reuses the lowest anonymous candidate ID; its frozen rating is not cherry-picked.',
                            'Intervals resample conversations conditional on frozen outputs and one correlated persona judge.']}
