"""Freeze full-data SEA English anchors and the paired renderer training snapshot."""
import hashlib,json,tarfile
from pathlib import Path
from replycoach.generate.sealion_full_rewrite_v1 import frozen_inputs
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
from replycoach.generate.sealion_rewrite_v1 import encode_rewrite,rewrite_source
from run_sealion_full_rewrite_v1 import pair_rows
ROOT=Path(__file__).resolve().parents[1]
DATA='data/training_expansion/generated600_repair_v1/combined_train.jsonl'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def rows(p):return [json.loads(l) for l in (ROOT/p).read_text().splitlines() if l.strip()]
def main():
    audit=read('data/training_expansion/sealion_rewrite_v1/pair_audit.json')
    if sha(ROOT/DATA)!=audit['source_sha256']:raise ValueError('Audited source changed')
    if not (ROOT/DATA).read_bytes().startswith((ROOT/'data/dev/train.jsonl').read_bytes()):raise ValueError('Team originals changed')
    full=read('artifacts/sealion_full_v1/results.json')
    evaluation=frozen_inputs(read('data/evaluation/sealion_rewrite_v1/inputs.json'),full)
    target=ROOT/'data/evaluation/sealion_full_rewrite_v1/inputs.json';target.parent.mkdir(exist_ok=True)
    if target.exists() and json.loads(target.read_text())!=evaluation:raise FileExistsError('Preserve different frozen inputs')
    target.write_text(json.dumps(evaluation,indent=2)+'\n')
    tok=checkpoint_tokenizer(Path('/tmp/sealion3b_source'));budgets={}
    for split,path in [('train',DATA),('validation','data/dev/validation.jsonl')]:
        ids={x['id'] for x in audit['items'] if x['split']==split and x['eligible']}
        encoded=[encode_rewrite(r,tok) for r in pair_rows(rows(path),ids)]
        budgets[split]=dict(count=len(encoded),max_tokens=max(len(x['input_ids']) for x in encoded))
    if budgets['train']['count']!=745 or budgets['validation']['count']!=42:raise ValueError('Pair counts')
    maxprompt=0;challenge=read('data/training_expansion/sealion_fidelity_v1/challenge.json')
    for cases,field in [(evaluation,'actual_anchor'),(evaluation,'diagnostic_anchor'),(challenge,'standard_anchor')]:
        for r in cases:
            if field not in r:continue
            n=len(tok.encode(rewrite_source(r,r[field]),add_special_tokens=False));maxprompt=max(maxprompt,n)
            if n+64>384:raise ValueError('Generation budget exceeded')
    vendor=read('src/replycoach/vendor/sealion_v1/provenance.json')
    files=[*sorted(ROOT.glob('src/**/*.py')),*[p for p in (ROOT/'src/replycoach/vendor/sealion_v1').iterdir() if p.is_file() and p.suffix!='.pyc'],*[ROOT/'scripts'/p for p in ['run_sealion_dual_v1.py','run_sealion_dialogue_v1.py','run_sealion_rewrite_v1.py','run_sealion_full_rewrite_v1.py','prepare_sealion_full_rewrite_v1.py','v5_sealion_full_rewrite_v1.sbatch']],*[ROOT/'tests'/p for p in ['test_sealion_full_rewrite_v1.py','test_sealion_rewrite_v1.py','test_sealion_v1_compat.py']],ROOT/DATA,ROOT/'data/dev/train.jsonl',ROOT/'data/dev/validation.jsonl',ROOT/'data/training_expansion/sealion_rewrite_v1/pair_audit.json',ROOT/'data/training_expansion/sealion_fidelity_v1/challenge.json',ROOT/'data/evaluation/sealion_rewrite_v1/inputs.json',target,ROOT/'artifacts/sealion_full_v1/results.json',ROOT/'artifacts/sealion_corrected_v1/s425.json']
    files=list(dict.fromkeys(files))
    recipe=dict(experiment='sealion_full_rewrite_v1',repo_id=vendor['repo_id'],revision=vendor['revision'],weights_sha256=vendor['weights']['lfs']['sha256'],english_generator_job=356177,english_generator_checkpoint='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_full_v1/artifacts/best_adapter',initial_adapter_sha256={'adapter_model.safetensors':'55a7f02626c08894bfdc4b44f68dacf86e8f4edafb7aa6bd78aead9162698913','adapter_config.json':'f5bb26bc63c376e64ed060f54cc87a44ece14a7328f064d02abc487aa75578bc'},initialization='Independent copy of full-data SEA356177 selected taskepoch2 adapter; original English adapter is never modified.',training=dict(seed=425,train_pairs=745,validation_pairs=42,learning_rate=3e-5,weight_decay=.01,microbatch=1,accumulation=8,max_epochs=8,patience=2,selection='Best reference-anchored LOCAL token CE among trainedepochs1+; initial baseline also reported.',loss='FP32 causal reply-only CE; conversation/draft masked; realEOS; no truncation.'),lora=dict(r=8,alpha=16,dropout=.05,trainable_parameters=10485760),evaluation=dict(actual_full_model_english=82,diagnostic_english=32,controlled_challenge=18,decoding='Greedy, maximum64 newtokens',baseline='Direct full-model LOCAL outputs paired with identical frozen English anchors; diagnostics/challenge are assistant authored, not human gold.',criteria=['Anchor fidelity: actors, polarity, times, amounts and decisions','Conversation usefulness when original English is useful','Natural Singlish without mandatory particles'],limitations='Reused development; reference anchors at training vs model-generated anchors in actual inference. English may already contain errors. Not an independent accuracy test or default promotion.'),budgets=budgets,max_generation_prompt_tokens=maxprompt,default_changed=False,english_generator_changed=False,files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    p=ROOT/'experiments/sealion_full_rewrite_v1_recipe.json';p.write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open('/tmp/replycoach_sealion_full_rewrite_v1.tgz','w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)))
        tar.add(p,arcname='recipe.json')
    print(json.dumps(dict(files=len(files),budgets=budgets,max_generation_prompt=maxprompt,actual_anchors=82,diagnostic=32,challenge=18)))
if __name__=='__main__':main()
