import hashlib,json,tarfile
from pathlib import Path
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
from replycoach.generate.sealion_rewrite_v1 import encode_rewrite,rewrite_source
ROOT=Path(__file__).resolve().parents[1]
def main():
    recipe=json.loads((ROOT/'experiments/sealion_rewrite_v1_recipe.json').read_text());recipe.update(experiment='sealion_fidelity_v1',initial_adapter_sha256={'adapter_model.safetensors':'b1f704d0fc3a547ef4eb05394bddada5fbf9e3b487c51905adf0e58ada3c3867','adapter_config.json':'625c7882c899de8cfc9d34b33037de2146a7980c253929d6038028f51e1acce9'})
    recipe['initialization']='Clone previous English-anchored renderer356079 selected epoch1, keep English generator unchanged.'
    recipe['training'].update(train_exposures_per_epoch=1490,replay_pairs=745,targeted_unique_pairs=86,targeted_exposures_per_epoch=745,validation_replay=42,validation_fidelity=18,selection='Equal mean of replay42 and targeted18 tokenCE, trained epochs1+, max8 patience2; challenge18 excluded from training and selection.')
    recipe['evaluation'].update(challenge=18,initial_challenge=18,limitations='Targeted examples are assistant-written controlled templates. Validation/challenge use separate template forms/time values/objects within new corpus, but may overlap historical/pretraining concepts. Not human gold or a blind generalization claim. Compare initial and selected renderer on identical heldout18; preserve old82+32 anchor tests.')
    files=[ROOT/p for p in recipe['files_sha256']]+[ROOT/'scripts/run_sealion_fidelity_v1.py',ROOT/'scripts/run_sealion_full_v1.py',ROOT/'scripts/run_sealion_expanded_v1.py',ROOT/'tests/test_sealion_parallel_v1.py',ROOT/'scripts/prepare_sealion_fidelity_data_v1.py',ROOT/'scripts/prepare_sealion_fidelity_run_v1.py',ROOT/'scripts/v5_sealion_fidelity_v1.sbatch',*list((ROOT/'data/training_expansion/sealion_fidelity_v1').glob('*.json'))]
    files=list(dict.fromkeys(files));tok=checkpoint_tokenizer(Path('/tmp/sealion3b_source'));budgets={}
    for split in ['train','validation','challenge']:
        rows=json.loads((ROOT/f'data/training_expansion/sealion_fidelity_v1/{split}.json').read_text());enc=[encode_rewrite(r,tok) for r in rows];budgets[split]=dict(count=len(enc),max_tokens=max(len(e['input_ids']) for e in enc))
        for r in rows:
            assert r['value'].lower() in r['standard_anchor'].lower() and r['value'].lower() in r['response'].lower()
            if r['kind']=='ownership':assert ('my '+r['object'] in r['response'].lower()) if r['direction']=='return_to_me' else ('your '+r['object'] in r['response'].lower())
    recipe['targeted_budgets']=budgets;recipe['files_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    out=ROOT/'experiments/sealion_fidelity_v1_recipe.json';out.write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open('/tmp/replycoach_sealion_fidelity_v1.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)))
        tar.add(out,arcname='recipe.json')
    print(json.dumps(dict(files=len(files),budgets=budgets)))
if __name__=='__main__':main()
