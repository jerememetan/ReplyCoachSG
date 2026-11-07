"""Evaluate frozen scorer revision on regression cases and unseen sampled instances."""
import hashlib,json,random
from pathlib import Path
from judge_replies import judge_settings,prepare_judge
from targeted_reward_v2 import PROMPT,score
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
assert hashlib.sha256(PROMPT.encode()).hexdigest()==recipe['scorer_prompt_sha256']
config=judge_settings({});settings=prepare_judge(config)
assert settings['model_digest']==recipe['expected_model_digest']
rows=[]
for partition,path in [('regression','data/reward_diagnostics/onpolicy_v1/labeled_samples.jsonl'),('fresh','data/reward_diagnostics/onpolicy_v2/labeled_samples.jsonl')]:
 rows += [dict(**json.loads(l),audit_partition=partition) for l in Path(path).read_text().splitlines()]
random.Random(426).shuffle(rows)
output=Path('artifacts/onpolicy2_reward.json');output.parent.mkdir(exist_ok=True)
if output.exists():raise FileExistsError('Preserve audit evidence')
report=dict(status='running',prompt=PROMPT,settings=settings,results=[],label_scope='Assistant judgments frozen before scorer calls; old regression set used for prompt development; fresh set not used for prompt edits')
def save():output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
try:
 for row in rows:
  report['results'].append({**row,**score(row,config,settings)});save();print('Scored',len(report['results']),'/96',flush=True)
 report['status']='complete';save()
except Exception:
 report['status']='failed';save();raise
