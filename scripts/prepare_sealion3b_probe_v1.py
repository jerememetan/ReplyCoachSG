"""Freeze the approved SEA-LION3B feasibility probe; no weights in the bundle."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    vendor=ROOT/'src/replycoach/vendor/sealion_v1'
    provenance=json.loads((vendor/'provenance.json').read_text())
    for name,digest in provenance['files_sha256'].items():
        if hashlib.sha256((vendor/name).read_bytes()).hexdigest()!=digest:raise ValueError('Upstream source changed: '+name)
    files=[ROOT/'src/replycoach/__init__.py',ROOT/'src/replycoach/schema.py',ROOT/'src/replycoach/tokenize.py',ROOT/'src/replycoach/generate/__init__.py',ROOT/'src/replycoach/vendor/__init__.py',
        *[ROOT/'src/replycoach/generate'/n for n in ['sealion_v1.py','sealion_data.py']],
        *[p for p in sorted(vendor.iterdir()) if p.is_file()],
        *[ROOT/'scripts'/n for n in ['prepare_sealion3b_probe_v1.py','download_sealion3b_probe_v1.py','run_sealion3b_probe_v1.py','v5_sealion3b_probe_v1.sbatch']],
        ROOT/'tests/test_sealion_data.py',ROOT/'tests/test_sealion_v1_compat.py',ROOT/'data/dev/train.jsonl']
    recipe=dict(experiment='sealion3b_probe_v1',repo_id=provenance['repo_id'],revision=provenance['revision'],weights_sha256=provenance['weights']['lfs']['sha256'],weights_bytes=provenance['weights']['size'],
        purpose='Feasibility only: original learned-position MPT architecture, real-weight strict loading, LoRA gradients/updates, peak VRAM, adapter save, short finite-logit autoregressive continuation.',
        gpu='One RTX3090,24GB',precision='BF16 backbone, PEFT LoRA adapters; no quantization or full-weight optimization',
        lora=dict(r=8,alpha=16,dropout=.05,target_modules=['Wqkv','out_proj','up_proj','down_proj']),
        probe=dict(steps=4,microbatch=1,padded_sequence_tokens=384,learning_rate=1e-4,seed=425,source='190 original training LOCAL rows only for budget checks;4 longest encoded rows for smoke steps; no validation/test access',masking='Prompt/pad labels -100; real target EOS retained even though pad and EOS IDs are shared'),
        compatibility=dict(upstream_code='Pinned reviewed source with MIT license, unchanged',transformers5='Add GenerationMixin and standard post_init loading bookkeeping; construct config explicitly with old tie_word_embeddings=True default. Preserve learned positions, non-ALiBi torch attention, and upstream forward/loss.',padding='Existing EOS reused for padding; no vocabulary resizing',inference='4 cache-free greedy steps only, not evaluated for reply quality'),
        environment='Existing personal Python3.13 venv reused read-only; einops0.8.1 installed only in this snapshot/deps, not shared mT5 env',
        next_step='Inspect actual checkpoint-loading correctness, gradient/update evidence and allocated/reserved VRAM. Passing only authorizes feasibility conclusion, not quality superiority or final-model promotion.',
        quality_evaluated=False,default_changed=False,
        files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    recipe_path=ROOT/'experiments/sealion3b_probe_v1_recipe.json';recipe_path.write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open('/tmp/replycoach_sealion3b_probe_v1.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)))
        tar.add(recipe_path,arcname='recipe.json')
    print('Pinned',recipe['repo_id'],recipe['revision'],'weights bytes',recipe['weights_bytes'],'source bundle files',len(files))

if __name__=='__main__':main()
