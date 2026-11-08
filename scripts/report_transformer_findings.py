"""Publish aggregate findings from the completed private transformer comparison."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import random
from statistics import mean

from replycoach.io import ROOT, read_jsonl
from replycoach.transformer_comparison import task_key


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def findings(folder):
    result, receipt = read(folder/'result.json'), read(folder/'judge_run.json')
    spec = importlib.util.spec_from_file_location('judge_report', ROOT/'scripts/judge_replies.py')
    judge = importlib.util.module_from_spec(spec); spec.loader.exec_module(judge)
    public, mapping, hashes = judge.load_packet(folder/'packet')
    if hashes != receipt['packet_hashes'] or checksum(folder/'freeze.json') != receipt['freeze_sha256']:
        raise ValueError('Source packet or frozen inputs changed')
    freeze = read(folder/'freeze.json')
    if (result.get('evaluation_split') != 'validation' or result.get('conversations') != 50
            or len(receipt['personas']) != 10 or len(public) != 300
            or any(r.get('evaluation_split') != 'validation' for r in mapping.values())
            or checksum(ROOT/'data/dev/validation.jsonl') != freeze['files']['data/dev/validation.jsonl']):
        raise ValueError('Expected the frozen 50-conversation validation comparison')
    expected_id = sha256(json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    ratings = read_jsonl(folder/'ratings.jsonl')
    latest = {}
    for row in ratings:
        key = row['candidate_id'], row['persona_id']
        if row['run_id'] != expected_id or key in latest:
            raise ValueError('Run identity mismatch or duplicate completed rating')
        if row['status'] != 'ok':
            raise ValueError('Report requires a complete, error-free canonical journal')
        judge.validate_rating(row['rating'])
        latest[key] = row
    expected = {(r['candidate_id'], p['id']) for r in public for p in receipt['personas']}
    if set(latest) != expected or len(latest) != 3000 or not result['completed']:
        raise ValueError('Expected 3000 completed candidate/persona rating cells')
    by_id = {r['candidate_id']: r for r in public}
    for (cid, persona), row in latest.items():
        if row['provenance']['kind'] == 'current_run':
            source = row['provenance']['candidate_id']
            if (task_key(by_id[cid], persona) != task_key(by_id[source], persona)
                    or row['rating'] != latest[source, persona]['rating']):
                raise ValueError('Reused rating is not an exact same-task/persona result')
    groups, cells = defaultdict(list), defaultdict(list)
    for (cid, _), row in latest.items():
        meta = mapping[cid]
        groups[meta['system'], meta['style']].append(row['rating'])
        cells[meta['system'], meta['item_id'], meta['style']].append(row['rating'])
    summary = []
    for old in result['summary']:
        rows = groups[old['model'], old['style']]
        if len(rows) != 500:
            raise ValueError('Every model/style needs the same 500 rating cells')
        values = {m: mean(r[m] for r in rows) for m in [*judge.SCORES, 'would_send']}
        if any(abs(values[m]-old[m]) > 1e-12 for m in values):
            raise ValueError('Published result differs from the journal')
        summary.append(dict(**old, rating_cells=len(rows),
            uncertain_count=sum(r['uncertain'] for r in rows),
            problem_counts={p: sum(r['problem']==p for r in rows) for p in judge.PROBLEMS},
            low_appropriateness_count=sum(r['appropriateness'] <= 2 for r in rows),
            score_distributions={m: {str(score):sum(r[m]==score for r in rows) for score in range(1,6)}
                                 for m in judge.SCORES}))
    items = sorted({m['item_id'] for m in mapping.values()})
    rng = random.Random(425)
    indices = [rng.choices(range(len(items)), k=len(items)) for _ in range(5000)]
    paired = []
    metrics = [*judge.SCORES, 'would_send', 'uncertain', 'problem_flagged']
    def metric_value(row, metric):
        return row['problem'] != 'none' if metric == 'problem_flagged' else row[metric]
    for other in ['Original mT5', 'New mT5 A']:
        for metric in metrics:
            delta = []
            for item in items:
                means = {}
                for model in ['SEA-LION 3B', other]:
                    means[model] = mean(mean(metric_value(r,metric) for r in cells[model,item,style])
                                        for style in ['STANDARD','LOCAL'])
                delta.append(means['SEA-LION 3B']-means[other])
            boot = sorted(sum(delta[i] for i in sample)/len(items) for sample in indices)
            paired.append(dict(top_model='SEA-LION 3B', other_model=other, metric=metric,
                difference=mean(delta), ci95=[boot[125],boot[4875]],
                units='score points' if metric in judge.SCORES else 'proportion'))
    sources = ['freeze.json','judge_run.json','result.json','ratings.jsonl',
               'packet/judge_inputs.jsonl','packet/PRIVATE_mapping.jsonl',
               'v5_outputs.json','mt5_handoff_outputs.json','sealion_handoff_outputs.json']
    return dict(schema_version=1, date='2026-11-08', completed=True, evaluation_split='validation',
        conversations=len(items), styles=2, personas=len(receipt['personas']), candidates=len(public),
        nominal_rating_cells=len(latest), provenance_counts=dict(Counter(r['provenance']['kind'] for r in ratings)),
        laptop_seed_ratings=result['cluster_runtime']['seeded_ratings'],
        fresh_cluster_ratings=sum(r.get('execution')=='cluster' and r['provenance']['kind']=='new_gemma_call' for r in ratings),
        summary=summary, overall=result['overall'], paired_all_metrics=paired,
        recommendation=result['recommended_model'], generation_failures=result['generation_failures'],
        generation_call_seconds=result['generation_latency'], judge=result['judge'],
        cluster_job_id=result['cluster_runtime']['job_id'], cluster_ollama_version=result['cluster_runtime']['ollama_version'],
        source_run=folder.relative_to(ROOT).as_posix(), source_sha256={n:checksum(folder/n) for n in sources},
        limitations=[result['limitation'], 'Ten personas are correlated views of one model.',
          '141 ratings originated on the laptop; remaining fresh ratings used the cluster. Same model digest and grading settings; cross-runtime replay was not performed.',
          'Forty exact-task rating reuses are not independent additional inferences.',
          'Generator-only selection: SEA-LION plus Qwen/V7 has not received a new matched quality evaluation.',
          'Pairwise 95% bootstrap intervals are unadjusted for multiple comparisons.'])


def markdown(report):
    lines = ['# Transformer selection findings — 8 November 2026', '',
        '**Decision:** use SEA-LION 3B for the interactive All models V5 slot. '
        'It leads on the predeclared primary criterion: mean appropriateness, with Standard and Local weighted equally.', '',
        '## Scope and method', '',
        '- Original V5: mT5 + LoRA (`artifacts/spk1_ul025_s425/best`).',
        '- New mT5 A: shared-intent rewrite, seed 1, final784 handoff.',
        '- SEA-LION 3B: English356177 + Singlish repair357651, step25 handoff.',
        '- Same 50 reviewed validation conversations, M2 memory and inferred generator speakers; no retrieval, Qwen selection or V7 refinement in this comparison.',
        '- 300 blinded candidates: 3 models × 50 conversations × 2 styles. Ten existing fictional personas give 3,000 complete rating cells (500 per model/style).',
        '- Existing Gemma4 12B model, rubric and criteria; temperature 0, seed 425, context 4096, prediction limit 2048, thinking disabled. No new grading criteria or training.',
        '- 2,960 actual fresh judgments and 40 verified exact-task/persona reuses. No older validation ratings were reused because their saved rubric differed.',
        '- 141 laptop ratings were retained; 2,819 fresh judgments ran on cluster L40S job 358003. The parallel laptop journal was backed up; overlapping work was not double-counted.',
        '- All three models generated all 50 reply pairs; zero generation failures and zero failed rating cells. Held-out test data was excluded from selection.', '',
        '## Every scored criterion, by style', '',
        'The three scores are on 1–5 scales. Would-send is simulated willingness with little or no editing, not actual user acceptance.', '',
        'Every rating also contains a short explanation. These rationales remain in the private raw journal; they are supporting text, not an additional numeric score.', '',
        '| Model | Style | Appropriateness | Naturalness | Style fit | Would send |',
        '|---|---|---:|---:|---:|---:|']
    for r in report['summary']:
        lines.append(f"| {r['model']} | {r['style']} | {r['appropriateness']:.3f} | {r['naturalness']:.3f} | {r['style_fit']:.3f} | {100*r['would_send']:.1f}% |")
    lines += ['', '## Overall (equal style weights)', '',
              '| Model | Appropriateness | Naturalness | Style fit | Would send |',
              '|---|---:|---:|---:|---:|']
    for m,r in report['overall'].items():
        lines.append(f"| {m} | {r['appropriateness']:.3f} | {r['naturalness']:.3f} | {r['style_fit']:.3f} | {100*r['would_send']:.1f}% |")
    lines += ['', '## Paired uncertainty for all metrics', '',
        'SEA-LION minus each competitor. Resample 50 conversations as paired blocks, retaining both styles and all ten personas together; 5,000 samples, seed 425. '
        'Intervals are descriptive, pairwise and unadjusted for multiple comparisons. Proportion differences below are percentage points; lower is better for uncertainty/problem flags.', '',
        '| Competitor | Metric | Difference | 95% bootstrap interval |', '|---|---|---:|---:|']
    for r in report['paired_all_metrics']:
        scale = 100 if r['units']=='proportion' else 1
        suffix = ' pp' if scale==100 else ''
        lines.append(f"| {r['other_model']} | {r['metric']} | {scale*r['difference']:+.3f}{suffix} | [{scale*r['ci95'][0]:+.3f}, {scale*r['ci95'][1]:+.3f}]{suffix} |")
    lines += ['', '## Problem and uncertainty flags', '',
        'Counts are out of 500 rating cells per model/style. `generic_or_unrelated_fallback` is the judge’s label; no generator failed or invoked a replacement generator in this run.', '',
        '| Model | Style | Generic/unrelated flag | Uncertain | Appropriateness ≤2 |', '|---|---|---:|---:|---:|']
    for r in report['summary']:
        lines.append(f"| {r['model']} | {r['style']} | {r['problem_counts']['generic_or_unrelated_fallback']} | {r['uncertain_count']} | {r['low_appropriateness_count']} |")
    lines += ['', 'All models had zero `garbled_or_unreadable`, `offensive_or_unsafe`, and `personal_info` flags. '
        'The aggregate JSON includes every problem category, including `none`, and complete 1–5 score distributions. '
        'These observed flags do not establish general safety or judge reliability.', '',
        '## Interpretation', '',
        '- SEA-LION improves overall appropriateness by +0.479 versus original mT5 and +0.865 versus new mT5 A. Both paired intervals stay above zero.',
        '- Simulated would-send increases from 70.5% (original) and 58.3% (new mT5 A) to 83.3% (SEA-LION).',
        '- Naturalness is nearly tied with original mT5 (4.961 versus 4.951); do not describe this tiny point difference as a demonstrated gain.',
        '- Standard style fit is 5.000 for every model, so it does not distinguish them. Local style-fit means are 3.896 original, 4.160 new mT5 A, and 4.206 SEA-LION; consult paired intervals before claiming style superiority.',
        '- New mT5 A raises Local style fit relative to original mT5 but lowers appropriateness and would-send in both styles. More recognisable local phrasing did not imply a better conversational answer.',
        '- The rubric explicitly does not establish Standard/Local meaning equivalence from separate single-reply grades; no paired semantic-equivalence claim is made.', '',
        '## Runtime and deployment boundary', '',
        'Recorded mean generator call times on the laptop were 0.804 s original, 1.665 s new mT5 A, and 3.833 s SEA-LION. '
        'These are diagnostics, not a controlled benchmark: handoff startup is counted inside the first call, while original V5 initialization occurs outside it. '
        'The cluster job completed in 2:30:43 including startup/download; an early 92-second parallel sample measured about 18.3 versus 9.8 ratings/minute.', '',
        'The current Streamlit All models V5 slot uses SEA-LION. The frozen final-test entry points continue explicitly loading original mT5. '
        'Historical V6/V7 results must not be relabelled as SEA-LION results. A live pipeline smoke check establishes integration, not new end-to-end quality evidence.', '',
        '## Limitations and evidence', '']
    lines.extend('- '+s for s in report['limitations'])
    lines += ['', 'The data is validation data used for model selection, and the candidates already have development histories. '
        'The 3,000 cells are not 3,000 independent conversations or human participants. '
        'Zero uncertainty flags in most groups do not prove the judge was certain or correct.', '',
        f"Private raw evidence: `{report['source_run']}/` (ignored). The adjacent aggregate JSON records input/result hashes, judge identity, reuse counts and full metric distributions without publishing conversations or credentials.", '',
        'Rebuild from the private evidence:', '', '```powershell',
        '.\\.venv\\Scripts\\python.exe scripts/report_transformer_findings.py --run experiments/judging/transformer_selection_20261108_01 --out docs/evaluation/transformer_selection_20261108', '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Aggregate report filename prefix')
    args = parser.parse_args()
    result = findings(args.run.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    args.out.with_suffix('.md').write_text(markdown(result), encoding='utf-8')
    print(json.dumps({'rating_cells':result['nominal_rating_cells'], 'recommendation':result['recommendation'],
                      'report':str(args.out.with_suffix('.md'))}))


if __name__ == '__main__':
    main()
