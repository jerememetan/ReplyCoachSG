"""Freeze matched parallel mT5 experiments from already reviewed public subset."""
import hashlib,io,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    original=Path('/tmp/replycoach_sealion_dialogue_v1.tar.gz');dest=Path('/tmp/replycoach_mt5_dialogue_v1.tar.gz')
    with tarfile.open(original,'r:gz') as src:
        files={m.name:src.extractfile(m).read() for m in src.getmembers() if m.isfile() and m.name!='recipe.json'}
    for name in ['scripts/run_mt5_dialogue_v1.py','scripts/prepare_mt5_dialogue_v1.py','scripts/v5_mt5_dialogue_v1.sbatch','tests/test_mt5_dialogue_v1.py','data/evaluation/mt5_dialogue_v1/reference.json']:
        files[name]=(ROOT/name).read_bytes()
    recipe=dict(experiment='mt5_dialogue_v1',arms=['public','control'],seeds=[425,426],base_checkpoint='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_standard_val_v1/artifacts/full_lr3e4/best',architecture='Full mT5-base encoder-decoder; direct STANDARD and LOCAL generation, all parameters trainable. No prompted LLM final model.',public=dict(pairs=7994,validation=400,max_passes=1,tag='REPLY for both corpora; no automatic corpus style labels',learning_rate=3e-5,selection='Equal-source public validation CE at 0/2000/4000/6000/end; zero public adaptation eligible; token overbudget exclusions recorded'),task=dict(pairs=380,validation=100,learning_rate=3e-5,max_epochs=8,patience=2,selection='Equal-style token CE; trained epochs1+ eligible',batch=4,accumulation=2),evaluation=dict(primary='Contextual usefulness and factual constraints; any valid reply intention acceptable',secondary='Naturalness, English/Singlish meaning alignment',new12='Frozen assistant-authored development cases, not human gold; same cases as staged SEA-LION',reused20='Previously inspected development; original mT5 English plus rewriter outputs frozen as baseline',reused_validation='50 original validation cases; not independent quality test',comparability='Public arm adds data/compute; not a matched-FLOP comparison. Both arms same task seeds/order/optimizer and original targets. Base checkpoint has historical generated/public training; these runs add no newly generated targets.'),default_changed=False,files_sha256={p:hashlib.sha256(b).hexdigest() for p,b in files.items()})
    raw=(json.dumps(recipe,indent=2)+'\n').encode();files['mt5_recipe.json']=raw
    (ROOT/'experiments/mt5_dialogue_v1_recipe.json').write_bytes(raw)
    with tarfile.open(dest,'w:gz') as out:
        for name,data in files.items():
            entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o600 if name.startswith('data/public/') else 0o644;out.addfile(entry,io.BytesIO(data))
    print(json.dumps(dict(archive=str(dest),files=len(files),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())))
if __name__=='__main__':main()
