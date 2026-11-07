"""Score a frozen sampled batch; no actor optimization or prompt changes."""
import hashlib,json,random
from pathlib import Path
from judge_replies import judge_settings,prepare_judge
from targeted_reward_v1 import PROMPT,score
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
assert hashlib.sha256(PROMPT.encode()).hexdigest()==recipe['scorer_prompt_sha256']
config=judge_settings({});settings=prepare_judge(config)
assert settings['model_digest']==recipe['expected_model_digest']
rows=[json.loads(l) for l in Path('data/reward_diagnostics/onpolicy_v1/labeled_samples.jsonl').read_text().splitlines()]
random.Random(425).shuffle(rows)
output=Path('artifacts/onpolicy_reward.json');output.parent.mkdir(exist_ok=True)
if output.exists():raise FileExistsError('Preserve audit output')
report=dict(status='running',prompt=PROMPT,settings=settings,results=[],source_sha256=recipe['files_sha256']['data/reward_diagnostics/onpolicy_v1/labeled_samples.jsonl'],label_scope='Assistant contextual judgments frozen before scorer calls; not human gold')
def save():output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
try:
 for row in rows:
  report['results'].append({**row,**score(row,config,settings)});save();print('Scored',len(report['results']),'/48',flush=True)
 report['status']='complete';save()
except Exception:
 report['status']='failed';save();raise
