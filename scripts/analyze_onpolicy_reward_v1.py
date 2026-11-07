"""Audit a frozen scorer on sampled replies; diagnostics, not human accuracy."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def summarize(rows,expected_count=48):
    definite=[r for r in rows if r['expected_verdict']!='uncertain']
    positive=[r for r in definite if r['expected_verdict']=='acceptable']
    negative=[r for r in definite if r['expected_verdict']=='unacceptable']
    def rate(group,label):return sum(r['rating']['verdict']==label for r in group)/len(group) if group else None
    accept=rate(positive,'acceptable');reject=rate(negative,'unacceptable')
    false_positive=[r['id'] for r in negative if r['rating']['verdict']=='acceptable']
    abstentions=sum(r['rating']['verdict']=='uncertain' for r in definite)
    groups={}
    for r in rows:
        g=groups.setdefault(r['item_id'],dict(category=r['category'],drafts=set(),rewards=[],expected=[]))
        g['drafts'].add(r['draft']);g['rewards'].append(r['rating']['reward']);g['expected'].append(r['expected_verdict'])
    mixed=sum(1 in g['rewards'] and -1 in g['rewards'] for g in groups.values())
    balance=(accept+reject)/2 if accept is not None and reject is not None else None
    gate=(len(rows)==expected_count and balance is not None and balance>=.9 and accept>=.8 and not false_positive and abstentions/max(1,len(definite))<=.2 and mixed>=3 and all('raw_failure' not in r for r in rows))
    return dict(samples=len(rows),scenario_groups=len(groups),unique_prompt_drafts=sum(len(g['drafts']) for g in groups.values()),expected_counts={k:sum(r['expected_verdict']==k for r in rows) for k in ('acceptable','unacceptable','uncertain')},positive_acceptance=accept,negative_rejection=reject,balanced_agreement=balance,false_positive_ids=false_positive,definite_abstentions=abstentions,mixed_reward_groups=mixed,raw_failures=sum('raw_failure' in r for r in rows),scoped_pilot_gate_pass=gate,scope='Assistant judgments on48 correlated samples from12 fresh instances, not human accuracy or broad scorer readiness',groups={k:dict(category=g['category'],unique_drafts=len(g['drafts']),rewards=g['rewards'],expected=g['expected']) for k,g in groups.items()})


def main():
    report=json.loads((ROOT/'artifacts/onpolicy_reward_v1/report.json').read_text())
    assert report['status']=='complete'
    rows=report['results'];summary=summarize(rows)
    disagreements=[r for r in rows if r['rating']['verdict']!=r['expected_verdict']]
    (ROOT/'experiments/onpolicy_reward_v1_results.json').write_text(json.dumps(dict(summary=summary,disagreements=disagreements),ensure_ascii=False,indent=2)+'\n')
    lines=['# Sampled policy reply reward audit','','No model training or scorer prompt edits during this audit. Labels are assistant contextual judgments frozen before scorer outcomes, not human gold.','','```json',json.dumps({k:v for k,v in summary.items() if k!='groups'},indent=2),'```','','## All sampled replies','','| Sample | Assistant verdict | Scorer verdict | Reply |','|---|---|---|---|']
    for r in sorted(rows,key=lambda x:x['id']):lines.append(f"| {r['id']} | {r['expected_verdict']} | {r['rating']['verdict']} | {r['draft'].replace('|','/')} |")
    lines+=['','## Disagreements','']
    for r in disagreements:
        lines.extend([f"### {r['id']}",f"Context: {r['context']}",f"Incoming: {r['message']}",f"Draft: {r['draft']}",f"Assistant judgment: {r['expected_verdict']}: {r['assessment_reason']}",f"Scorer: {r['rating']['verdict']}: {r['rating']['reason']}",''])
    (ROOT/'experiments/onpolicy_reward_v1.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
