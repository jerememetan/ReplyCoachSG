"""Freeze corrected600 + exact original190, two SEA-LION task seeds."""
import copy,hashlib,json,tarfile
from pathlib import Path
from run_sealion_expanded_v1 import expanded_rows
from replycoach.generate.sealion_data import encode_reply
from replycoach.generate.sealion_dual_v1 import item_rows
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
ROOT=Path(__file__).resolve().parents[1]
DATA='data/training_expansion/generated600_repair_v1'
def rows(p):return [json.loads(l) for l in (ROOT/p).read_text().splitlines() if l.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    folder=ROOT/DATA;manifest=json.loads((folder/'manifest.json').read_text())
    for p,h in manifest['files_sha256'].items():
        if sha(folder/p)!=h:raise ValueError('Corrected data audit hash mismatch '+p)
    if sha(ROOT/'data/dev/train.jsonl')!=manifest['sources']['team_sha256'] or sha(ROOT/'data/dev/validation.jsonl')!=manifest['sources']['validation_sha256']:raise ValueError('Original protected source changed')
    if not (folder/'combined_train.jsonl').read_bytes().startswith((ROOT/'data/dev/train.jsonl').read_bytes()):raise ValueError('Team bytes changed')
    original=rows('data/dev/train.jsonl');val=rows('data/dev/validation.jsonl');training=rows(DATA+'/combined_train.jsonl')
    tok=checkpoint_tokenizer(Path('/tmp/sealion3b_source'))
    encoded=[encode_reply(r,tok,384) for r in expanded_rows(training,original,val,'preserved')];valid=[encode_reply(r,tok,384) for r in item_rows(val)]
    files=[*sorted(ROOT.glob('src/**/*.py')),*[p for p in (ROOT/'src/replycoach/vendor/sealion_v1').iterdir() if p.is_file()],*[ROOT/'scripts'/p for p in ['run_sealion_expanded_v1.py','run_sealion_dual_v1.py','run_sealion_dialogue_v1.py','prepare_sealion_corrected_v1.py','build_generated600_repair_v1.py','human_expansion_review.py','v5_sealion_expanded_v1.sbatch']],*[ROOT/'tests'/p for p in ['test_sealion_expanded_v1.py','test_generated600_repair_v1.py','test_human_expansion_review.py','test_sealion_data.py','test_sealion_dual_v1.py','test_sealion_v1_compat.py']],ROOT/'data/dev/train.jsonl',ROOT/'data/dev/validation.jsonl',ROOT/'data/training_expansion/blanket600_v1/expanded.jsonl',ROOT/'data/evaluation/sealion_dual_v1/inputs.json',ROOT/'data/evaluation/sealion_dialogue_v1/inputs.json',*[p for p in folder.iterdir() if p.is_file()]]
    files=list(dict.fromkeys(files));recipe=json.loads((ROOT/'experiments/sealion_expanded_v1_recipe.json').read_text())
    recipe.update(experiment='sealion_corrected_v1',arms=['corrected'],seeds=[425,426],training_file=DATA+'/combined_train.jsonl',require_original_team_exact=True)
    recipe['data'].update(policies=dict(preserved='Exact original190 team records, unchanged byte-for-byte, plus600 assistant-reviewed generated additions'),approval='User authorized assistant correction of600 generated additions and training. Not per-item human gold ratings.',quality='515 generated examples revised,85 unchanged;195 English/502 LOCAL targets changed,16 generated conversation/message repairs. See full frozen correction audit.',correction_manifest_sha256=sha(folder/'manifest.json'),combined_train_sha256=sha(folder/'combined_train.jsonl'))
    recipe['evaluation']['comparison']='Seed425 matches previous preserved790 run356050 initialization, task RNG/order/optimizer/validation; corrected additions are intervention (includes16 input repairs). Seed426 additional robustness replicate, no matched old426 control. All82 evaluation conversations reused development; not independent human gold. Evaluate usefulness, uncertainty calibration, natural Singlish and paired meaning, not just CE.'
    recipe['budgets']=dict(train_targets=len(encoded),max_train_tokens=max(len(r['input_ids']) for r in encoded),validation_targets=len(valid),truncated=False)
    recipe['files_sha256']={str(p.relative_to(ROOT)):sha(p) for p in files}
    for seed in [425,426]:
        current=copy.deepcopy(recipe);current['training']['seed']=seed
        p=ROOT/f'experiments/sealion_corrected_v1_s{seed}_recipe.json';p.write_text(json.dumps(current,indent=2)+'\n')
        with tarfile.open(f'/tmp/replycoach_sealion_corrected_v1_s{seed}.tar.gz','w:gz') as tar:
            for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)))
            tar.add(p,arcname='recipe.json')
    print(json.dumps(dict(files=len(files),seeds=[425,426],budgets=recipe['budgets'])))
if __name__=='__main__':main()
