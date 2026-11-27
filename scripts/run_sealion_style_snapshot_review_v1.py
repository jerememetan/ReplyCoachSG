"""Review trained snapshots even when conservative CE selection retains step0."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_style_v1 as runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['control','copy_loss'],required=True)
    args=parser.parse_args()
    assert Path.home().name=='hm.teoh.2024'
    recipe=runner.read('recipe.json')
    for name,expected in recipe['files_sha256'].items():assert runner.sha(ROOT/name)==expected,name
    progress=runner.read('outputs/training/'+args.arm+'/training.json')
    assert progress['status']=='trained'
    rows=runner.read('data/evaluation.json');controls=runner.read('data/controls.json')
    ids=['SEADUAL-011','SEADUAL-015','SEADUAL-008','SEADUAL-010','SEADUAL-013']
    chosen=[row for ident in ids for row in rows if row['item_id']==ident]
    remaining=[row for row in rows if row['item_id'] not in ids]
    chosen+=random.Random(20261108).sample(remaining,17)
    chosen+=controls
    runner.LEXICON=runner.style.fit_lexicon(runner.read('data/train.json'))
    runner.torch.set_num_threads(4)
    tok=runner.checkpoint_tokenizer(runner.BUNDLE/'base')
    destination=ROOT/'outputs/snapshot_review'/args.arm
    destination.mkdir(parents=True,exist_ok=False)
    for step in [25,50,100]:
        checkpoint=ROOT/'outputs/training'/args.arm/f'step_{step:04d}'
        model=runner.load(checkpoint).eval()
        report=dict(status='running',arm=args.arm,step=step,selected_step=progress['selected_step'],
            checkpoint=str(checkpoint),checkpoint_sha256=runner.sha(checkpoint/'adapter_model.safetensors'),
            scope='Exploratory32-case snapshot review, including previously inspected5. Nonselected snapshots are not approved upgrades.',
            default_changed=False,human_review_pending=True)
        runner.evaluate(model,tok,chosen,destination/f'step_{step:04d}.json',report,expanded=False)
        del model;runner.clear()
    runner.save(destination/'summary.json',dict(status='complete',arm=args.arm,checkpoints=[25,50,100],
        cases_per_checkpoint=len(chosen),selected_step=progress['selected_step'],default_changed=False))


if __name__=='__main__':main()
