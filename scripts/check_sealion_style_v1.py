"""Cluster-only data/loader preflight; no updates to model weights."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
BUNDLE=Path.home()/'ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1'
sys.path[:0]=[str(ROOT),str(BUNDLE/'src'),str(BUNDLE/'deps')]
import sealion_style_v1 as style
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
from replycoach.generate.sealion_rewrite_v1 import encode_rewrite

for name,expected in json.loads((ROOT/'preflight_files.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
spec=importlib.util.spec_from_file_location('export_loader',BUNDLE/'infer.py')
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
manifest=loader.verify_files(BUNDLE)
assert manifest['files']['base/model.safetensors']['sha256']=='acba6d6560f61ac72cc95feca5c89f6aabbbf1e9e89c18440237331fcb5ae107'
tokenizer=checkpoint_tokenizer(BUNDLE/'base')
sets={name:json.loads((ROOT/'data'/f'{name}.json').read_text()) for name in ['train','validation','copy_train','copy_validation','evaluation','controls']}
assert len(sets['train'])==745 and len(sets['validation'])==42
assert not {r['item_id'] for r in sets['train']} & {r['item_id'] for r in sets['validation']+sets['evaluation']}
assert all(style.auxiliary_eligible(row) for split in ['copy_train','copy_validation'] for row in sets[split])
budgets={}
for split in ['train','validation','copy_train','copy_validation']:
    maximum=0
    for row in sets[split]:
        variants=[row,dict(row,response=row['standard_anchor'])] if split.startswith('copy_') else [row]
        for item in variants:
            encoded=encode_rewrite(item,tokenizer,384)
            maximum=max(maximum,len(encoded['input_ids']))
            assert all(x==-100 for x in encoded['labels'][:encoded['prompt_tokens']])
            assert encoded['labels'][-1]==tokenizer.eos_token_id
    budgets[split]=maximum
assert all(len(tokenizer.encode(row['standard_anchor'],add_special_tokens=False))>0 for row in sets['evaluation']+sets['controls'])
report=dict(status='passed',counts={k:len(v) for k,v in sets.items()},max_tokens=budgets,
    split_overlap=False,source_bundle_checksums_verified=True,
    source_manifest_sha256=hashlib.sha256((BUNDLE/'manifest.json').read_bytes()).hexdigest(),
    source_weights_sha256={k:v['sha256'] for k,v in manifest['files'].items() if k.endswith('.safetensors')},
    training_performed=False)
(ROOT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
