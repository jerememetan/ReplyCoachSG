"""Consolidate fit diagnostics without equating target memorization with product accuracy."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/fit_probe_v1/artifacts'
NAMES=('lora_lr1e4','lora_lr3e4','full_lr1e4','full_lr3e4')
def analyze():
 rows=[json.loads(l) for l in (ROOT/'data/training_diagnostics/fit_probe_v1/train.jsonl').read_text().splitlines()]
 paras=[json.loads(l) for l in (ROOT/'data/training_diagnostics/fit_probe_v1/paraphrases.jsonl').read_text().splitlines()]
 assert len(rows)==len(paras)==24 and {x['paired_train_id'] for x in paras}=={x['item_id'] for x in rows}
 assert not {x['item_id'] for x in rows}&{x['item_id'] for x in paras}
 train_targets={x['item_id']:x['response'] for x in rows};assert all(x['response']==train_targets[x['paired_train_id']] for x in paras)
 summary=json.loads((ROOT/'artifacts/fit_probe_v1/review/summary.json').read_text());baseline=json.loads((ART/'base.json').read_text());runs={}
 for name in NAMES:
  path=ART/f'{name}.json';r=json.loads(path.read_text())
  assert r['status']=='complete' and r['optimizer_steps']==400 and r['train_items']==24 and r['training_style']=='STANDARD' and r['batch_size']==4
  assert r['base_files_sha256']==baseline['base_files_sha256']
  assert [x['optimizer_steps'] for x in r['history']]==[0,100,400]
  assert len(r['paraphrase_generations'])==24 and all(len(r['train_generations'][str(step)])==24 for step in [0,100,400])
  if r['method']=='full':assert r['trainable_parameters']==r['total_parameters']==582401280
  else:assert r['trainable_parameters']==1769472
  runs[name]=dict(method=r['method'],learning_rate=r['learning_rate'],optimizer_steps=r['optimizer_steps'],trainable_parameters=r['trainable_parameters'],total_parameters=r['total_parameters'],history=r['history'],paraphrase_exact_matches=r['paraphrase_exact_matches'],paraphrase_reference_loss=r['paraphrase_reference_loss'],peak_cuda_gb=r['peak_cuda_gb'],epochs_started=r['epochs_started'],checkpoint=r['checkpoint'],report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),training_adequacy=summary['counts']['train'][name],paraphrase_adequacy=summary['counts']['paraphrase'][name])
 result=dict(status='complete',runs=runs,baseline=dict(train=summary['counts']['train']['base'],paraphrase=summary['counts']['paraphrase']['base']),quality_review=summary,decision='Model can learn these STANDARD targets: both full runs exactly fit24/24 by100 updates and retain24/24 at400; both reach18/24 paraphrase adequacy. LoRA improves to21/24 training and15/24 paraphrases at3e-4 but falls short of22/24 training gate under tested settings. Restricted adaptation/optimization is a candidate bottleneck, alongside remaining wording transfer errors. This does not prove LoRA cannot eventually fit, establish the larger-task cause or justify production replacement.',limitations=['Single seed,24 chosen scenarios,14 assistant-checked targets and10 user-approved; not human gold for all cases.','Paraphrases share underlying scenarios/entities; this measures wording transfer, not broad unseen-scenario reasoning.','Two fixed learning rates and400 updates do not establish optimal LoRA or full-finetuning hyperparameters.','Both arms are STANDARD-only; this does not isolate STANDARD/LOCAL interference or establish retained LOCAL style.','Exact reference match measures target adherence; adequacy includes valid alternatives and uncertain judgments.','Reference matches98/240 auto-annotated;142 alternatives inspected by the assistant; not240 independent human labels.','No original held-out test reads or production default change.'])
 (ROOT/'experiments/fit_probe_v1_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('Verified4 completed400-update runs, byte-identical starting checkpoints and all240 ratings.')
if __name__=='__main__':analyze()
