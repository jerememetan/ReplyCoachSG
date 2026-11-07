"""Summarize every held-out preference and explicitly labeled assistant audit."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'artifacts/human_dpo_v1'
report=json.loads((folder/'report.json').read_text())
pairs=json.loads((folder/'heldout_pair_evaluation.json').read_text())['pairs']
# Contextual assistant review of ALL20 fresh prompts; these are not human gold.
judgments=[
('fail','fail','Agrees to5pm despite cannot leave until6:30.'),
('fail','fail','Reverses ownership/roles: peer wants to keep the user calculator; user replies they can keep it.'),
('fail','fail','Claims report finished although user has not started.'),
('uncertain','uncertain','Sympathy is plausible but fails to clearly acknowledge waiting; weak rather than an explicit contradiction.'),
('fail','fail','Contradictory yes/closed answer and invents bakery opening status despite admitted uncertainty.'),
('fail','fail','Agrees to Sunday despite user unavailable Sunday.'),
('fail','fail','Answers where the bottle was lost instead of its colour.'),
('pass','fail','Before confirms completed upload, with incidental timing; DPO instead promises future upload despite it already being done.'),
('pass','pass','Agrees to return the borrowed umbrella; correct owner and role.'),
('fail','fail','Accepts twenty-dollar buffet despite twelve-dollar budget.'),
('pass','pass','Correctly states west entrance from context.'),
('fail','fail','Offers to teach software despite user not knowing how to use it.'),
('pass','pass','Availability after four supports meeting at4:30.'),
('fail','fail','Before self-contradicts and changes topic to arrival; after unexplained refusal and new booking promise do not resolve extra guest.'),
('fail','uncertain','Before chooses cookies user cannot eat; DPO acknowledges inability but wraps it in an awkward preference. Partial repair, no confident pass.'),
('pass','pass','Correctly confirms possession of peer headphones.'),
('fail','fail','Agrees to nine despite needing to leave by eight.'),
('pass','pass','Offers to check unknown deadline rather than invent one.'),
('pass','pass','Acknowledges thanks and confirms poster commitment.'),
('pass','pass','Correctly states tickets not bought yet.'),
]
base={r['item_id']:r for r in report['baseline_generations']};dpo={r['item_id']:r for r in report['dpo_generations']}
fresh=[]
for n,(before,after,note) in enumerate(judgments,1):
 item=f'FRESH-DPO-{n:03}'
 fresh.append(dict(item_id=item,baseline=before,dpo=after,note=note,baseline_reply=base[item]['generated_standard'],dpo_reply=dpo[item]['generated_standard']))
counts={arm:{label:sum(r[arm]==label for r in fresh) for label in ('pass','fail','uncertain')} for arm in ('baseline','dpo')}
summary=dict(preference_pairs=len(pairs),conversations=len({r['item_id'] for r in pairs}),baseline_human_choice_ranked_higher=sum(r['baseline']['margin']>0 for r in pairs),dpo_human_choice_ranked_higher=sum(r['dpo']['margin']>0 for r in pairs),corrected=sum(r['baseline']['margin']<=0<r['dpo']['margin'] for r in pairs),regressed=sum(r['dpo']['margin']<=0<r['baseline']['margin'] for r in pairs),relative_margin_improved=sum(r['margin_change']>0 for r in pairs),mean_validation_dpo_loss=sum(r['dpo_loss'] for r in pairs)/len(pairs),fresh_assistant_review=counts,limitation='17 correlated preference pairs from8 conversations, previously used in SFT selection/human review. No locked test accessed. Sequence probability ranking is not generated reply quality. Fresh20 audit is post-hoc assistant assessment, not blinded human gold. No statistical superiority claim.')
(folder/'full_evaluation.json').write_text(json.dumps(dict(summary=summary,pairs=pairs,fresh_assistant_review=fresh),ensure_ascii=False,indent=2)+'\n')
lines=['# Full DPO pilot evaluation','',json.dumps(summary,indent=2),'','## All17 held-out human preferences','','Positive margin means the model gives the user-chosen reply a higher total sequence log probability. Delta is DPO minus pre-DPO margin.','','| # | Conversation | Baseline margin | DPO margin | Delta |','|---|---|---:|---:|---:|']
for n,r in enumerate(pairs,1):lines.append(f"| {n} | {r['item_id']} | {r['baseline']['margin']:.3f} | {r['dpo']['margin']:.3f} | {r['margin_change']:+.3f} |")
lines+=['','## All20 fresh conversations: assistant review','','Rubric: relevance to latest message, correct user/peer roles, compatibility with explicit context facts/constraints. Pass allows alternative wording and ordinary selectable personal drafts; uncertain flags borderline responses. No reference text matching.','','| Case | Baseline | DPO | Finding |','|---|---|---|---|']
for r in fresh:lines.append(f"| {r['item_id']} | {r['baseline']} | {r['dpo']} | {r['note']} |")
lines+=['','## Interpretation','','No demonstrated generalization gain from this pilot. The selected DPO checkpoint improves training preference loss, but held-out sequence ranking does not improve. Fresh contextual inspection reveals persistent role/constraint failures, one clear upload regression and one partial peanut-response repair. This is evidence to collect more varied preference/correction data or improve context conditioning, not to increase epochs blindly. Model remains available for user testing; no automatic acceptance/rejection or deployment change.']
(ROOT/'experiments/human_dpo_v1_full_evaluation.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
