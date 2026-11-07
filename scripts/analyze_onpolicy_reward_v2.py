"""Keep revised-scorer development comparisons separate from its fresh audit."""
import json
from pathlib import Path
from analyze_onpolicy_reward_v1 import summarize
ROOT=Path(__file__).resolve().parents[1]


def analyze(report,old_report):
    if report['status']!='complete':raise ValueError('Full batch required')
    fresh=[r for r in report['results'] if r['audit_partition']=='fresh']
    regression=[r for r in report['results'] if r['audit_partition']=='regression']
    if len(fresh)!=48 or len(regression)!=48:raise ValueError('Expected48 fresh and48 regression replies')
    if {r['item_id'] for r in fresh}&{r['item_id'] for r in regression}:raise ValueError('Fresh group overlap')
    old={r['id']:r for r in old_report['results']}
    repaired=[];new_errors=[]
    for r in regression:
        before=old[r['id']]
        for k in ('draft','context','message','expected_verdict'):assert r[k]==before[k],k
        was_correct=before['rating']['verdict']==r['expected_verdict'];is_correct=r['rating']['verdict']==r['expected_verdict']
        if not was_correct and is_correct:repaired.append(r['id'])
        if was_correct and not is_correct:new_errors.append(r['id'])
    return dict(fresh=summarize(fresh),regression_v1=summarize(old_report['results']),regression_v2=summarize(regression),regression_repaired_ids=repaired,regression_new_error_ids=new_errors,limitation='Regression cases informed v2 prompt, not independent validation. Fresh48 replies are correlated within12 new same-family instances. Labels are assistant judgments, not human gold. Fresh gate unchanged, no post-result relabeling or prompt edits.')


def main():
    report=json.loads((ROOT/'artifacts/onpolicy_reward_v2/report.json').read_text());old=json.loads((ROOT/'artifacts/onpolicy_reward_v1/report.json').read_text())
    results=analyze(report,old)
    (ROOT/'experiments/onpolicy_reward_v2_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    lines=['# Revised reward scorer audit v2','','## Summary','','```json',json.dumps({k:v for k,v in results.items() if k not in ('fresh','regression_v1','regression_v2')},indent=2),'```','']
    for name in ('fresh','regression_v1','regression_v2'):
        lines += [f'## {name}','','```json',json.dumps({k:v for k,v in results[name].items() if k!='groups'},indent=2),'```','']
    for partition in ('fresh','regression'):
        lines += [f'## All {partition} replies','','| Sample | Assistant verdict | Scorer verdict | Reply |','|---|---|---|---|']
        for r in sorted(report['results'],key=lambda r:r['id']):
            if r['audit_partition']==partition:lines.append(f"| {r['id']} | {r['expected_verdict']} | {r['rating']['verdict']} | {r['draft'].replace('|','/')} |")
        lines+=['','### Disagreements','']
        for r in report['results']:
            if r['audit_partition']==partition and r['rating']['verdict']!=r['expected_verdict']:
                lines += [f"**{r['id']}** · Context {r['context']} · Incoming: {r['message']}",f"Draft: {r['draft']}",f"Assistant: {r['expected_verdict']} — {r['assessment_reason']}",f"Scorer: {r['rating']['verdict']} — {r['rating']['reason']}",'']
    (ROOT/'experiments/onpolicy_reward_v2.md').write_text('\n'.join(lines)+'\n')
    for name in ('fresh','regression_v1','regression_v2'):print(name,json.dumps({k:v for k,v in results[name].items() if k!='groups'}))
    print('Repaired',results['regression_repaired_ids'],'new errors',results['regression_new_error_ids'])

if __name__=='__main__':main()
