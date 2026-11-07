"""Freeze larger adaptation as a separate intervention from fidelity training."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    folder=ROOT/'data/public/sealion_full_v1';m=json.loads((folder/'manifest.json').read_text())
    if m['status']!='ready_exploratory':raise ValueError('Sample inspection required')
    for p,h in m['files_sha256'].items():
        if sha(folder/p)!=h:raise ValueError('Full data hash mismatch')
    vendor=json.loads((ROOT/'src/replycoach/vendor/sealion_v1/provenance.json').read_text());data=ROOT/'data/training_expansion/generated600_repair_v1'
    files=[*sorted(ROOT.glob('src/**/*.py')),*[p for p in (ROOT/'src/replycoach/vendor/sealion_v1').iterdir() if p.is_file()],*[ROOT/'scripts'/p for p in ['run_sealion_dual_v1.py','run_sealion_dialogue_v1.py','run_sealion_expanded_v1.py','run_sealion_full_v1.py','prepare_sealion_full_public_v1.py','prepare_sealion_full_run_v1.py','v5_sealion_full_v1.sbatch']],*[p for p in folder.iterdir() if p.is_file()],*[p for p in data.iterdir() if p.is_file()],ROOT/'data/dev/train.jsonl',ROOT/'data/dev/validation.jsonl',ROOT/'data/evaluation/sealion_dual_v1/inputs.json',ROOT/'data/evaluation/sealion_dialogue_v1/inputs.json',ROOT/'tests/test_sealion_parallel_v1.py',*[p for p in (ROOT/'data/training_expansion/sealion_fidelity_v1').glob('*.json')]]
    files=list(dict.fromkeys(files));recipe=dict(experiment='sealion_full_v1',repo_id=vendor['repo_id'],revision=vendor['revision'],weights_sha256=vendor['weights']['lfs']['sha256'],public=dict(train_pairs=m['train_rows'],cosem=53691,dailydialog=63943,validation=400,passes=1,initialization='Fresh LoRA r8/a16/dropout.05 on original pinned SEA-LION base, same initialization policy as earlier small public run; not continuing from already-task-trained model.',learning_rate=5e-5,microbatch=1,accumulation=16,tag='DIALOGUE for both corpora',selection='Use final full-pass checkpoint for task phase so it has seen all usable pairs. Public validation monitors every10000 but does not revert to an earlier checkpoint.',checkpoint='Atomic trusted local .pt of LoRA weights, optimizer, RNG, progress and report every4000 examples, only at optimizer boundaries; restart same script to resume interrupted public pass.'),task=dict(conversations=790,targets=1580,validation=100,seed=425,learning_rate=1e-4,microbatch=1,accumulation=8,max_epochs=8,patience=2,selection='Minimum original100 target equal-style CE among trained epochs1+. Same task dataset/recipe as corrected subset-public run356064, public size+compute are intervention.'),evaluation=dict(reused12=12,reused20=20,reused_validation=50,scope='Reused development, not independent human gold or locked test. Compare English usefulness and LOCAL fidelity/naturalness against356064; do not mix this intervention into fidelity renderer pilot.',limitations=m['limitations']),runtime=dict(gpu='L40S',walltime_hours=12,system_memory_gb=32),default_changed=False,files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    p=ROOT/'experiments/sealion_full_v1_recipe.json';p.write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open('/tmp/replycoach_sealion_full_v1.tar.gz','w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)))
        tar.add(p,arcname='recipe.json')
    print(json.dumps(dict(files=len(files),public=m['train_rows'],cosem=53691,dailydialog=63943)))
if __name__=='__main__':main()
