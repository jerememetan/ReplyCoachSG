"""Freeze one anchored-renderer pilot and its explicit paired-data eligibility."""
import hashlib,json,tarfile
from pathlib import Path
from run_sealion_rewrite_v1 import pair_rows
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
from replycoach.generate.sealion_rewrite_v1 import encode_rewrite,rewrite_source
ROOT=Path(__file__).resolve().parents[1]
DATA='data/training_expansion/generated600_repair_v1/combined_train.jsonl'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in (ROOT/p).read_text().splitlines()]
def main():
    audit=json.loads((ROOT/'data/training_expansion/sealion_rewrite_v1/pair_audit.json').read_text())
    if sha(ROOT/DATA)!=audit['source_sha256']:raise ValueError('Audited source changed')
    if not (ROOT/DATA).read_bytes().startswith((ROOT/'data/dev/train.jsonl').read_bytes()):raise ValueError('Team data changed')
    tok=checkpoint_tokenizer(Path('/tmp/sealion3b_source'));budgets={}
    for split,path in [('train',DATA),('validation','data/dev/validation.jsonl')]:
        ids={x['id'] for x in audit['items'] if x['split']==split and x['eligible']};encoded=[encode_rewrite(r,tok) for r in pair_rows(rows(path),ids)]
        budgets[split]=dict(count=len(encoded),max_tokens=max(len(x['input_ids']) for x in encoded))
    evaluation=json.loads((ROOT/'data/evaluation/sealion_rewrite_v1/inputs.json').read_text())
    max_prompt=0
    for r in evaluation:
        for field in ['actual_anchor','diagnostic_anchor']:
            if field in r:
                n=len(tok.encode(rewrite_source(r,r[field]),add_special_tokens=False));max_prompt=max(max_prompt,n)
                if n+64>384:raise ValueError('Generation budget exceeded')
    vendor=json.loads((ROOT/'src/replycoach/vendor/sealion_v1/provenance.json').read_text())
    files=[*sorted(ROOT.glob('src/**/*.py')),*[p for p in (ROOT/'src/replycoach/vendor/sealion_v1').iterdir() if p.is_file()],*[ROOT/'scripts'/p for p in ['run_sealion_dual_v1.py','run_sealion_dialogue_v1.py','run_sealion_rewrite_v1.py','prepare_sealion_rewrite_v1.py','prepare_sealion_rewrite_audit_v1.py','v5_sealion_rewrite_v1.sbatch']],*[ROOT/'tests'/p for p in ['test_sealion_rewrite_v1.py','test_sealion_data.py','test_sealion_v1_compat.py']],ROOT/DATA,ROOT/'data/dev/train.jsonl',ROOT/'data/dev/validation.jsonl',ROOT/'data/training_expansion/sealion_rewrite_v1/pair_audit.json',ROOT/'data/evaluation/sealion_rewrite_v1/inputs.json',ROOT/'artifacts/sealion_corrected_v1/s425.json']
    files=list(dict.fromkeys(files));recipe=dict(experiment='sealion_rewrite_v1',repo_id=vendor['repo_id'],revision=vendor['revision'],weights_sha256=vendor['weights']['lfs']['sha256'],initial_adapter_sha256={'adapter_model.safetensors':'12afa8a6b8456d5608f5871b7cdb5b93488cd011d2becf512f79572ff2040aa1','adapter_config.json':'9900705ae33a671ade8db19444bcb819fe8285379c7e0df90e9997f00d1cf224'},initialization='Clone selected corrected-data SEA-LION seed425 epoch2 LoRA into new rewriting adapter; never overwrite/change original English/direct adapter. Same pinned BF16 base.',pair_audit=audit['counts'],training=dict(seed=425,train_pairs=745,validation_pairs=42,learning_rate=3e-5,weight_decay=.01,microbatch=1,accumulation=8,max_epochs=8,patience=2,selection='Minimum42 reference-anchored LOCAL validation tokenCE among trained epochs1+. No selection on sampled outputs; CE under different conditioning/subset is not comparable with previous direct50 CE.',loss='FP32 reply-only causal CE; conversation and English anchor fully masked, real EOS retained, no truncation.'),lora=dict(r=8,alpha=16,dropout=.05,trainable_parameters=10485760),evaluation=dict(actual_english=82,diagnostic_english=32,baseline='Frozen seed425 direct English/LOCAL results from356064 on identical12+20+50 inputs; English generator never fine-tuned here.',diagnostic='32 explicitly assistant-authored context-compatible English anchors; reused development, not independent human gold, never training targets. Actual English may contradict context; separate anchor fidelity from contextual correctness. No exact reference-intent requirement or particle-count reward.',criteria=['Does rewritten LOCAL preserve English actors, polarity, amounts, times and decision?','Does it remain context-compatible where anchor is compatible?','Is it natural Singlish according to user judgement?','Does English-first pipeline improve over direct LOCAL on same frozen English outputs?'],limitations='Teacher English anchors at training differ from generated English at inference; route mismatch explicitly measured. Conservative assistant exclusions can be imperfect. Pilot one seed; no reliable quality claim or automatic app promotion.'),budgets=budgets,max_generation_prompt_tokens=max_prompt,default_changed=False,files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    p=ROOT/'experiments/sealion_rewrite_v1_recipe.json';p.write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open('/tmp/replycoach_sealion_rewrite_v1.tar.gz','w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)))
        tar.add(p,arcname='recipe.json')
    print(json.dumps(dict(files=len(files),budgets=budgets,max_generation_prompt=max_prompt)))
if __name__=='__main__':main()
