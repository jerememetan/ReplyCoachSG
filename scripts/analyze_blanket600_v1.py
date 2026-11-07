"""Verify all-candidate training and preserve evaluation/provenance evidence."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DIRECTORY=ROOT/'artifacts/blanket600_v1/artifacts'
def analyze():
 manifest=json.loads((ROOT/'data/training_expansion/blanket600_v1/manifest.json').read_text());ledger=json.loads((ROOT/'data/training_expansion/blanket600_v1/approval_ledger.json').read_text())
 assert len(ledger)==600 and sum(x['approval']['individually_reviewed'] for x in ledger)==10
 summary=json.loads((ROOT/'artifacts/blanket600_v1/review/summary.json').read_text());runs={}
 for arm,items,rows,context in [('control',190,560,180),('expanded',790,2226,646)]:
  path=DIRECTORY/f'{arm}_s425.json';r=json.loads(path.read_text());p=json.loads((DIRECTORY/f'{arm}_preflight.json').read_text())
  assert (r['optimizer_steps'],r['seed'],r['train_items'],r['train_pairs'],r['train_pairs_by_kind']['context_pairs'])==(1120,425,items,rows,context)
  assert r['rewrite_local'] and r['lora']['targets']==['q','v'] and r['trainable_parameters']==1769472
  assert r['preferences'] is None and r['unlikelihood'] is None
  outputs={}
  for checkpoint,steps in [('final',1120),('step_784',784)]:
   f=DIRECTORY/f'{arm}_{checkpoint}_outputs.json';x=json.loads(f.read_text())
   assert x['optimizer_steps']==steps and len(x['validation_outputs'])==50 and len(x['diagnostic_outputs'])==24 and x['preflight']==p
   assert len({c['item_id'] for c in x['validation_outputs']})==50 and len({c['item_id'] for c in x['diagnostic_outputs']})==24
   outputs[checkpoint]=dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),exact_style_copies_validation=sum(c['generated_standard'].strip().casefold().rstrip('.!')==c['generated_local'].strip().casefold().rstrip('.!') for c in x['validation_outputs']),exact_style_copies_diagnostic=sum(c['generated_standard'].strip().casefold().rstrip('.!')==c['generated_local'].strip().casefold().rstrip('.!') for c in x['diagnostic_outputs']))
  runs[arm]=dict(report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),preflight=p,optimizer_steps=r['optimizer_steps'],epochs_completed=r['history'][-1]['epoch'],trainable_parameters=r['trainable_parameters'],peak_cuda_gb=r['peak_cuda_gb'],best_validation_loss=r['best_validation_loss'],best_epoch=r['best_epoch'],final_losses={k:v for k,v in r['history'][-1].items() if 'loss' in k},training_validation_seconds=r['history'][-1]['seconds'],outputs=outputs)
 result=dict(status='complete',runs=runs,quality_review=summary,authorization=dict(individually_reviewed=10,blanket_authorized=590,user_instruction=manifest['user_instruction']),decision='Training completed; do not replace default. STANDARD validation adequacy is36/50 both arms, while Singlish falls38 to25 and all-four falls28 to16 with expansion. Diagnostic all-four improves3 to7 but STANDARD falls12 to9. No consistent general quality upgrade established.',limitation='One seed, reused50+24 development cases and assistant quality judgments. Extra466 auxiliary context rows are part of expansion. Equal1120 updates are32 control epochs versus8 expanded epochs, not equal exposure or FLOPs. Blanket authorization is not item-level human preference data. Step784 and best-by-loss retained secondary; no post-hoc checkpoint substitution or locked-test score.')
 (ROOT/'experiments/blanket600_v1_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('Verified2 completed1120-step runs,790 expanded conversations and all evaluation artifacts.')
 for arm,r in runs.items():print(arm,r['outputs']['final'])
if __name__=='__main__':analyze()
