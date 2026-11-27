"""Inference-only check of all three methods on experimental copy-loss100."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_style_v1 as frozen
import sealion_style_guard_v2 as revised


def main():
    assert Path.home().name=='hm.teoh.2024'
    recipe=frozen.read('recipe.json')
    for name,expected in recipe['files_sha256'].items():assert frozen.sha(ROOT/name)==expected,name
    assert frozen.read('outputs/training/results.json')['status']=='complete'
    previous=frozen.read('outputs/snapshot_review/copy_loss/step_0100.json')
    assert previous['status']=='complete'
    cases=[{key:row[key] for key in ['item_id','scope','standard_anchor','context','speakers','message']} for row in previous['outputs']]
    assert len(cases)==32
    checkpoint=ROOT/'outputs/training/copy_loss/step_0100'
    assert frozen.sha(checkpoint/'adapter_model.safetensors')==previous['checkpoint_sha256']
    frozen.torch.set_num_threads(4)
    frozen.LEXICON=frozen.style.fit_lexicon(frozen.read('data/train.json'))
    # Explicit runtime strategy injection in this new process only. revised.BASE
    # is an independent module, so its frozen guard is not patched recursively.
    frozen.style.guard=revised.guard
    tokenizer=frozen.checkpoint_tokenizer(frozen.BUNDLE/'base')
    model=frozen.load(checkpoint).eval()
    report=dict(status='running',checkpoint=str(checkpoint),step=100,
        checkpoint_sha256=previous['checkpoint_sha256'],guard='contractions_v2',
        original_validation_selected_step=0,training_performed=False,default_changed=False,
        scope='32 reused review cases; exploratory nonselected copy-loss100 plus shared guards/reranking/grammar. Human assessment pending.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    folder=ROOT/'outputs/combined_copy100_v2';assert not folder.exists();folder.mkdir()
    frozen.evaluate(model,tokenizer,cases,folder/'results.json',report,expanded=True)
    prior={row['item_id']:row for row in previous['outputs']}
    report['raw_checkpoint_replay_differences']=[row['item_id'] for row in report['outputs']
        if row['methods']['beam4']['text']!=prior[row['item_id']]['methods']['beam4']['text']]
    assert not report['raw_checkpoint_replay_differences']
    report['source_adapters_unchanged']=all(frozen.sha(frozen.BUNDLE/name)==value for name,value in frozen.read('preflight.json')['source_weights_sha256'].items() if name.startswith('adapters/'))
    assert report['source_adapters_unchanged']
    frozen.save(folder/'results.json',report)
    print(json.dumps(dict(status='complete',cases=len(cases),replay_differences=0,source_adapters_unchanged=True)),flush=True)


if __name__=='__main__':main()
