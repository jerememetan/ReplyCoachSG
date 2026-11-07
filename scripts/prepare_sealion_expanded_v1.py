"""Freeze two target-policy arms of the authorized 790-conversation experiment."""
import hashlib,json,tarfile
from pathlib import Path
from run_sealion_expanded_v1 import expanded_rows
from replycoach.generate.sealion_data import encode_reply
from replycoach.generate.sealion_dual_v1 import item_rows
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
ROOT=Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(l) for l in (ROOT/p).read_text().splitlines()]
def signature(r):return json.dumps([[t['speaker'],t['text'].strip().lower()] for t in r['session_memory']]+[['peer',r['message']['text'].strip().lower()]])
def main():
    original=rows('data/dev/train.jsonl');val=rows('data/dev/validation.jsonl');expanded=rows('data/training_expansion/blanket600_v1/expanded.jsonl')
    if {signature(r) for r in expanded}&{signature(r) for r in val}:raise ValueError('Exact validation input overlap')
    tok=checkpoint_tokenizer(Path('/tmp/sealion3b_source'));budgets={}
    for policy in ['saved','preserved']:
        data=expanded_rows(expanded,original,val,policy);encoded=[encode_reply(r,tok,384) for r in data]
        budgets[policy]=dict(train_targets=len(encoded),max_train_tokens=max(len(e['input_ids']) for e in encoded))
    valid=[encode_reply(r,tok,384) for r in item_rows(val)]
    vendor=json.loads((ROOT/'src/replycoach/vendor/sealion_v1/provenance.json').read_text())
    files=[*sorted(ROOT.glob('src/**/*.py')),*[p for p in (ROOT/'src/replycoach/vendor/sealion_v1').iterdir() if p.is_file()],*[ROOT/'scripts'/p for p in ['run_sealion_expanded_v1.py','prepare_sealion_expanded_v1.py','run_sealion_dual_v1.py','run_sealion_dialogue_v1.py','v5_sealion_expanded_v1.sbatch']],*[ROOT/'tests'/p for p in ['test_sealion_expanded_v1.py','test_sealion_data.py','test_sealion_dual_v1.py','test_sealion_v1_compat.py']],ROOT/'data/dev/train.jsonl',ROOT/'data/dev/validation.jsonl',*[ROOT/'data/training_expansion/blanket600_v1'/p for p in ['expanded.jsonl','manifest.json','approval_ledger.json']],ROOT/'data/evaluation/sealion_dual_v1/inputs.json',ROOT/'data/evaluation/sealion_dialogue_v1/inputs.json']
    files=list(dict.fromkeys(files))
    recipe=dict(experiment='sealion_expanded_v1',arms=['saved','preserved'],repo_id=vendor['repo_id'],revision=vendor['revision'],weights_sha256=vendor['weights']['lfs']['sha256'],public_adapter_sha256='8636d9ca1bcf6ba74572fbf76aeddd058dc805818956a67527b41ca885384555',initialization='Same selected public-only LoRA checkpoint after7994 pairs from355937; excludes subsequent190 task training. Frozen base reused; no public retraining.',data=dict(train_conversations=790,original=190,additions=600,train_targets=1580,validation_conversations=50,validation_targets=100,policies=dict(saved='Exact saved expanded.jsonl target pairs, including14 revised original English and190 revised original LOCAL targets',preserved='Original190 exact paired targets restored; same600 additions retained unchanged'),approval='10 additions individually reviewed,590 previously blanket authorized; generated targets not independent human gold. No new generated targets or silent corrections added.',quality='Small exploratory inspection found unsupported details in some additions (e.g. group size); both arms preserve additions to measure the authorized dataset. Larger data is not assumed better.'),training=dict(seed=425,learning_rate=1e-4,weight_decay=.01,microbatch=1,accumulation=8,max_epochs=8,patience=2,selection='Lowest equal-style original validation CE among trained epochs1+; no output-based checkpoint selection',precision='BF16 frozen base; FP32 reply-only causal CE',sequence_limit=384),lora=dict(r=8,alpha=16,dropout=.05,target_modules=['Wqkv','out_proj','up_proj','down_proj']),evaluation=dict(reused12=12,reused20=20,reused_validation=50,primary='Context-compatible useful replies, then natural Singlish and cross-style meaning; no exact-intent requirement or particle-count reward',comparison='Compare both policies with existing original190 public-adapted SEA355937 outputs; quantity+training compute differ, old190 RNG details not exactly matched. Saved/preserved arms otherwise identical seeds/order/initialization/optimizer. All evaluation reused development, not blind human gold or locked test.',default_changed=False),budgets=budgets,files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    p=ROOT/'experiments/sealion_expanded_v1_recipe.json';p.write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open('/tmp/replycoach_sealion_expanded_v1.tar.gz','w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)))
        tar.add(p,arcname='recipe.json')
    print(json.dumps(dict(files=len(files),budgets=budgets,validation=len(valid))))
if __name__=='__main__':main()
