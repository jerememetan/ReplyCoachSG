"""Consolidate frozen reports without equating token loss with reply accuracy."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'artifacts/standard_focus_v1/artifacts'

def analyze():
    runs={}
    for path in sorted(DIRECTORY.glob('*_s*.json')):
        if path.stem.endswith('_diagnostic'):continue
        r=json.loads(path.read_text());arm,seed=path.stem.rsplit('_s',1)
        if arm not in ('control','targeted','capacity','standard_heavy'):continue
        diag_path=DIRECTORY/f'{path.stem}_diagnostic.json';diag=json.loads(diag_path.read_text())
        assert r['optimizer_steps']==784 and r['seed']==int(seed)
        assert r['train_items']==(190 if arm=='control' else 270)
        assert r['train_pairs_by_kind']['context_pairs']==180
        assert len(r['final_validation_outputs'])==50 and len(diag['outputs'])==24
        assert len({x['item_id'] for x in diag['outputs']})==24
        assert r['train_pairs']=={'control':560,'targeted':720,'capacity':720,'standard_heavy':990}[arm]
        assert r['rewrite_local'] and r['speaker_labels']
        runs[path.stem]=dict(seed=int(seed),arm=arm,optimizer_steps=r['optimizer_steps'],train_items=r['train_items'],train_rows=r['train_pairs'],trainable_parameters=r['trainable_parameters'],best_validation_loss=r['best_validation_loss'],best_epoch=r['best_epoch'],final_losses={k:v for k,v in r['history'][-1].items() if 'loss' in k},training_seconds=r['history'][-1]['seconds'],peak_cuda_gb=r.get('peak_cuda_gb'),report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),diagnostic_sha256=hashlib.sha256(diag_path.read_bytes()).hexdigest())
    reviews={}
    for dirname in ('review','review_s1','review_s2'):
        path=ROOT/'artifacts/standard_focus_v1'/dirname/'summary.json'
        if path.exists():reviews[dirname]=json.loads(path.read_text())
    result=dict(runs=runs,reviews=reviews,limitation='Assistant judgments, not independent human accuracy. Fixed final-step primary; all seeds repeat the same50 existing and24 newly authored development conversations. Candidate selected after seed425; replication is not a new held-out set. Oracle-conditioned LOCAL reference loss is not independent generation quality. No default replacement or final-test access.')
    (ROOT/'experiments/standard_focus_v1_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Verified',len(runs),'completed runs;',len(reviews),'completed review packets.')
if __name__=='__main__':analyze()
