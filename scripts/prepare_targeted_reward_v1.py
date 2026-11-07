"""Freeze human policy calibration, untouched instance diagnostics and actual outputs."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/reward_diagnostics/targeted_rl_v1'
out=ROOT/'data/reward_diagnostics/targeted_scorer_v1'
if out.exists():raise FileExistsError('Freeze once before scorer outcomes')
review=json.loads((source/'human_reviews.json').read_text());rows=[]
for split in ('review','development','heldout'):
    for r in map(json.loads,(source/f'{split}.jsonl').read_text().splitlines()):
        expected=[review[r['id']]['verdict']] if split=='review' else [r['proposed_verdict']]
        provenance='explicit human verdict' if split=='review' else 'assistant proposed diagnostic label'
        # Policy transfer declared before scoring; original proposed labels remain in row.
        if split!='review' and r['draft']=='Let me check and get back to you.':
            expected=['acceptable'];provenance='policy-derived from two explicit human approvals of same deferral; not per-item human approval'
        if split!='review' and r['draft']=='Maybe, I will let you know.':
            expected=['uncertain','unacceptable'];provenance='both human judgments preserved; positive reward forbidden'
        rows.append({**r,'expected_verdicts':expected,'expectation_provenance':provenance})
actual=json.loads((ROOT/'artifacts/human_dpo_v1/full_evaluation.json').read_text())['fresh_assistant_review']
contexts={r['item_id']:r for r in map(json.loads,(ROOT/'data/training_preferences/human_dpo_v1/fresh_audit.jsonl').read_text().splitlines())}
seen={};labels={'pass':'acceptable','fail':'unacceptable','uncertain':'uncertain'}
category={1:'time',2:'ownership',3:'completion',4:'coordination',5:'knowledge',6:'availability',7:'non_answer',8:'completion',9:'ownership',10:'budget',11:'location',12:'ability',13:'availability',14:'coordination',15:'constraint',16:'ownership',17:'time',18:'knowledge',19:'commitment',20:'completion'}
for r in actual:
    c=contexts[r['item_id']];number=int(r['item_id'].rsplit('-',1)[1])
    for arm in ('baseline','dpo'):
        text=r[f'{arm}_reply'];signature=(r['item_id'],text)
        if signature in seen:
            assert seen[signature]['expected_verdicts']==[labels[r[arm]]]
            seen[signature]['sources'].append(arm);continue
        row=dict(id=f"ACTOR-{r['item_id']}-{arm}",scenario_id=r['item_id'],split='actor',category=category[number],context=c['context'],speakers=c['speakers'],message=c['message'],draft=text,expected_verdicts=[labels[r[arm]]],expectation_provenance='prior contextual assistant review of saved model output; not human gold',sources=[arm])
        seen[signature]=row;rows.append(row)
out.mkdir(parents=True)
(out/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(out/'manifest.json').write_text(json.dumps(dict(cases=len(rows),partition_counts={s:sum(r['split']==s for r in rows) for s in ('review','development','heldout','actor')},label_policy='Scorer sees only conversation, incoming message, style and draft. Original labels and policy transfer declared before outcomes.',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'human_reviews.json',source/'review.jsonl',source/'development.jsonl',source/'heldout.jsonl',ROOT/'artifacts/human_dpo_v1/full_evaluation.json']}),indent=2)+'\n')
print(len(rows),'cases frozen')
