import hashlib,json,random
from pathlib import Path
from judge_replies import judge_settings,prepare_judge
from targeted_reward_v1 import PROMPT,score,summarize
recipe=json.loads(Path('recipe.json').read_text())
for f,digest in recipe['files_sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
config=judge_settings({});settings=prepare_judge(config)
assert settings['model_digest']==recipe['expected_model_digest'],'Cached model digest differs from frozen recipe'
rows=[json.loads(l) for l in Path('data/reward_diagnostics/targeted_scorer_v1/cases.jsonl').read_text().splitlines()]
random.Random(425).shuffle(rows)
output=Path('artifacts/targeted_reward.json')
if output.exists():raise FileExistsError('Preserve completed and failed scorer evidence')
report=dict(status='running',settings=settings,prompt=PROMPT,results=[],scope='Human policy calibration and assistant-labeled small diagnostics, no RL updates')
def save():output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
try:
    for row in rows:
        report['results'].append({**row,**score(row,config,settings)});save()
        print('Scored',len(report['results']),'/',len(rows),flush=True)
    report.update(status='complete',summary=summarize(report['results']));save();print(json.dumps(report['summary']),flush=True)
except Exception:
    report['status']='failed';save();raise
