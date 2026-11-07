"""Check isolated critic verdicts against explicitly reviewed single drafts."""
import argparse
from hashlib import sha256
from pathlib import Path

from compare_critic_modes import SINGLE_RUBRIC, SINGLE_SCHEMA, SINGLE_VERSION, judge_single
from critic_preferences import save
from judge_replies import judge_settings, prepare_judge, read_env
from replycoach.io import ROOT, read_jsonl


def validate(cases):
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Unique nonempty reviewed cases required')
    for c in cases:
        if (type(c.get('acceptable')) is not bool or not c.get('reviewer')
                or not c.get('review_source') or not isinstance(c.get('draft'), str)
                or not c['draft'].strip() or c.get('style') not in ('STANDARD', 'LOCAL')
                or not isinstance(c.get('conversation'), list) or not isinstance(c.get('message'), str)):
            raise ValueError('Reviewed single draft, input and boolean label required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh report')
    cases = read_jsonl(args.cases)
    validate(cases)
    config = judge_settings(read_env(args.env_file)); settings = prepare_judge(config)
    report = dict(purpose='reviewed_critic_diagnostics_only', status='running', settings=settings,
                  prompt_version=SINGLE_VERSION, rubric=SINGLE_RUBRIC, schema=SINGLE_SCHEMA,
                  cases_sha256=sha256(args.cases.read_bytes()).hexdigest(), results=[],
                  limitation='Small user-reviewed development checks used to clarify policy; not an independent benchmark or training preferences')
    save(args.output, report)
    try:
        for c in cases:
            result = judge_single(c, c['draft'], config, settings)
            expected = 'acceptable' if c['acceptable'] else 'unacceptable'
            report['results'].append({**c, 'judgment': result, 'correct': result['verdict'] == expected})
            save(args.output, report)
            print(c['id'], result['verdict'], result['verdict'] == expected, flush=True)
    except Exception:
        report['status'] = 'failed'; save(args.output, report); raise
    report['status'] = 'complete'
    report['summary'] = dict(cases=len(cases), correct=sum(r['correct'] for r in report['results']),
        incorrect_acceptance=sum(not r['acceptable'] and r['judgment']['verdict']=='acceptable' for r in report['results']),
        incorrect_rejection=sum(r['acceptable'] and r['judgment']['verdict']=='unacceptable' for r in report['results']),
        uncertain=sum(r['judgment']['verdict']=='uncertain' for r in report['results']))
    save(args.output, report)
    print(report['summary'])


if __name__ == '__main__':
    main()
