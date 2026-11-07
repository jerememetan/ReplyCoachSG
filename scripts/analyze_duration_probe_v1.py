"""Verify all snapshot artifacts and retain full duration results without loss-based winner claims."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DIRECTORY=ROOT/'artifacts/duration_probe_v1/artifacts'
STEPS=(160,320,480,784)
def analyze():
 source=ROOT/'data/reward_diagnostics/duration_probe_v1.jsonl';digest=hashlib.sha256(source.read_bytes()).hexdigest()
 ids={json.loads(l)['id'] for l in source.read_text().splitlines()};runs={};snapshots={};reproduction={}
 for arm in ('control','capacity'):
  report_path=DIRECTORY/f'{arm}_s425.json';r=json.loads(report_path.read_text())
  assert r['optimizer_steps']==784 and set(r['step_checkpoints'])==set(map(str,STEPS))
  assert r['train_pairs']==(560 if arm=='control' else 720) and r['train_pairs_by_kind']['context_pairs']==180
  assert r['seed']==425 and r['rewrite_local'] and r['speaker_labels']
  runs[arm]=dict(optimizer_steps=r['optimizer_steps'],train_rows=r['train_pairs'],trainable_parameters=r['trainable_parameters'],history=r['history'],report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
  old=json.loads((ROOT/'artifacts/standard_focus_v1/artifacts'/f'{arm}_s425.json').read_text())
  previous={c['item_id']:(c['generated_standard'],c['generated_local']) for c in old['final_validation_outputs']}
  reproduction[arm]=dict(prior_run_identical_in_memory_final_pairs=sum((c['generated_standard'],c['generated_local'])==previous[c['item_id']] for c in r['final_validation_outputs']),total=50,prior_reference_loss=old['history'][-1]['validation_loss_standard'],current_reference_loss=r['history'][-1]['validation_loss_standard'])
  for step in STEPS:
   p=DIRECTORY/f'{arm}_step{step}.json';x=json.loads(p.read_text())
   assert x['fresh_source_sha256']==digest and x['optimizer_step']==step and x['arm']==arm and x['seed']==425
   assert len(x['outputs'])==24 and {c['item_id'] for c in x['outputs']}==ids and len(x['secondary_validation_outputs'])==50
   snapshots[f'{arm}_step{step}']=dict(standard_reference_loss=x['original_reference_standard_loss'],fresh_pairs=24,secondary_pairs=50,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 review=json.loads((ROOT/'artifacts/duration_probe_v1/review/summary.json').read_text())
 assert set(review['counts'])==set(snapshots)
 audit=json.loads((DIRECTORY/'control_reload_audit.json').read_text());assert audit['step_and_final_tensors_equal'] and all(audit['metadata_equal'].values())
 result=dict(status='complete',runs=runs,snapshots=snapshots,quality_review=review,prior_run_reproduction=reproduction,reload_audit=audit,decision='No early-duration quality upgrade established. Both final variants have12/24 STANDARD passes; best earlier control has11 and broader candidate has12. Broader all-four is9 at480 and784, versus6 at320 despite lower reference loss. Keep default unchanged; no extra training or locked-test run.',limitation='One seed on24 assistant-authored development contexts,192 correlated checkpoint pairs, not192 independent conversations or human accuracy. Original50 generations are retained secondary evidence, not fully quality-rated here. Original control verification rejected text differences on identical weights; inference-only recovery preserves raw failure and48/50 matching secondary pairs.')
 (ROOT/'experiments/duration_probe_v1_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('Verified2 training trajectories,8 snapshots and192 complete pair ratings.');print(json.dumps(reproduction,indent=2))
if __name__=='__main__':analyze()
