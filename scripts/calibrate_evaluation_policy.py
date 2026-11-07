"""Check evaluation rubric against six user-reviewed drafts; never create training labels."""
import argparse,json
from hashlib import sha256
from pathlib import Path

from critic_preferences import save
from judge_replies import judge_settings,prepare_judge,read_env,judge_one,load_personas
from replycoach.io import ROOT,read_jsonl


def policy_matches(expected,rating):
    return (not rating['uncertain'] and rating['would_send']==expected
            and (rating['appropriateness']>=4 if expected else rating['appropriateness']<=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cases',type=Path);p.add_argument('--rubric',type=Path,required=True)
    p.add_argument('--personas',type=Path,default=ROOT/'data/evaluation/personas.csv')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--env-file',type=Path,default=ROOT/'.env')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new diagnostic report')
    cases=read_jsonl(args.cases)
    if len(cases)!=6 or len({c['id'] for c in cases})!=6 or any(type(c.get('acceptable')) is not bool for c in cases):
        raise ValueError('Six explicitly reviewed single drafts required')
    config=judge_settings(read_env(args.env_file));settings=prepare_judge(config)
    persona=load_personas(args.personas)[0];rubric=args.rubric.read_text()
    report=dict(purpose='evaluation_policy_alignment_diagnostic_only',status='running',settings=settings,
        persona=persona,rubric=rubric,rubric_sha256=sha256(args.rubric.read_bytes()).hexdigest(),
        cases_sha256=sha256(args.cases.read_bytes()).hexdigest(),results=[],
        gate='Expected acceptable: appropriateness >=4 and would_send true; unacceptable: <=2 and false; uncertainty fails gate.',
        limitation='Six policy-development checks, not independent accuracy or the full ten-persona evaluation; never training labels.')
    save(args.output,report)
    try:
        for c in cases:
            candidate={'conversation':[{'speaker':s,'text':t} for s,t in c['conversation']],
                'incoming_message':{'speaker':'peer','text':c['message']},'style':c['style'],'reply':c['draft']}
            rating,runtime=judge_one(config['base'],settings['judge_model'],persona,candidate,rubric)
            result={**c,'rating':rating,'runtime':runtime,'policy_match':policy_matches(c['acceptable'],rating)}
            report['results'].append(result);save(args.output,report)
            print(c['id'],rating['appropriateness'],rating['would_send'],result['policy_match'],flush=True)
    except Exception:
        report['status']='failed';save(args.output,report);raise
    report.update(status='complete',summary={'cases':6,'policy_matches':sum(r['policy_match'] for r in report['results'])})
    save(args.output,report);print(report['summary'])


if __name__=='__main__':main()
