"""Run preserved mT5 example references without changing any checkpoint."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer
import run_mt5_adapter_switch_v1 as core
from run_mt5_context_pairs_v1 import generate

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',choices=['historical_final','historical_best','current_shared','all'],default='historical_final')
    parser.add_argument('--inputs',type=Path,default=ROOT/'experiments/mt5_reference_pack_v1/common_validation50.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-common50',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous outputs')
    manifest=json.loads((ROOT/'experiments/mt5_reference_pack_v1/manifest.json').read_text())
    base=Path(manifest['common_base']['source_path'])
    for name,h in manifest['common_base']['files_sha256'].items():
        if sha(base/name)!=h:raise ValueError('Adapted base identity changed: '+name)
    core.TOK=AutoTokenizer.from_pretrained(base,local_files_only=True)
    cases=json.loads(args.inputs.read_text())
    if len(cases)!=len({r['item_id'] for r in cases}):raise ValueError('Duplicate IDs')
    if args.verify_common50:
        expected=json.loads((ROOT/'experiments/mt5_reference_pack_v1/common_validation50.json').read_text())
        if cases!=expected:raise ValueError('Verification requires exact frozen50 cases')
    result=dict(status='running',checkpoint_unchanged=True,models={},training_performed=False,
                input_sha256=sha(args.inputs),decode=dict(num_beams=4,no_repeat_ngram_size=3,max_new_tokens=48,max_source=192))
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    save();torch.set_num_threads(4)
    for name in manifest['models'] if args.reference=='all' else [args.reference]:
        spec=manifest['models'][name]; adapter=Path(spec['checkpoint'])
        for filename,h in [('adapter_model.safetensors',spec['adapter_sha256']),('adapter_config.json',spec['config_sha256'])]:
            if sha(adapter/filename)!=h:raise ValueError('Preserved adapter identity changed')
        model=PeftModel.from_pretrained(core.load_base(base),adapter,is_trainable=False).eval()
        batch_size=4 if name=='current_shared' else 50
        english=generate(model,cases,'STANDARD','beam4',batch_size=batch_size)
        local=generate(model,cases,'LOCAL_REWRITE','beam4',[r['text'] for r in english],batch_size=batch_size)
        outputs=[];differences=[]
        for row,e,l in zip(cases,english,local,strict=True):
            outputs.append(dict(item_id=row['item_id'],english=e,local=l,standard_anchor=e['text']))
            if args.verify_common50 and (e['text']!=row['models'][name]['english'] or l['text']!=row['models'][name]['local']):
                differences.append(dict(item_id=row['item_id'],expected=row['models'][name],actual=dict(english=e['text'],local=l['text'])))
        result['models'][name]=dict(checkpoint=str(adapter),outputs=outputs,
             exact_saved_pairs=len(cases)-len(differences) if args.verify_common50 else None,differences=differences)
        save();print(name,'pairs',len(cases),'differences',len(differences),flush=True)
        for filename,h in [('adapter_model.safetensors',spec['adapter_sha256']),('adapter_config.json',spec['config_sha256'])]:
            if sha(adapter/filename)!=h:raise ValueError('Adapter mutated during inference')
        del model;gc.collect();torch.cuda.empty_cache()
    result['status']='complete' if all(not r['differences'] for r in result['models'].values()) else 'complete_with_replay_differences'
    save()


if __name__=='__main__':main()
