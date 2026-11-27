"""Run the optional guarded pipeline on custom inputs or frozen cached baselines."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parent
BUNDLE=Path.home()/'ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1'
sys.path[:0]=[str(ROOT),str(BUNDLE/'src'),str(BUNDLE/'deps')]
import torch
import sealion_context_guard_v1 as policy
import sealion_context_pipeline_v1 as guarded


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=ROOT/'inputs.json')
    parser.add_argument('--cached-baselines',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'results.json')
    args=parser.parse_args()
    assert Path.home().name=='hm.teoh.2024'
    assert not args.output.exists()
    torch.set_num_threads(4)
    manifest=json.loads((ROOT/'payload_manifest.json').read_text())
    assert all(sha(ROOT/name)==value for name,value in manifest.items())
    import replycoach.generate
    sys.modules['replycoach.generate.sealion_context_guard_v1']=policy
    sys.modules['replycoach.generate.sealion_context_pipeline_v1']=guarded
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_sealion_context_guard_v1.py')
    checked=unittest.TextTestRunner(verbosity=1).run(suite)
    assert checked.wasSuccessful() and checked.testsRun==31
    spec=importlib.util.spec_from_file_location('original_bundle_infer',BUNDLE/'infer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=json.loads(args.inputs.read_text());module.validate_inputs(rows)
    started=time.monotonic()
    original=module.ReplyPipeline(BUNDLE,device='cuda')
    pipeline=guarded.ContextGuardedPipeline(original)
    cached=json.loads(args.cached_baselines.read_text()) if args.cached_baselines else []
    indexed={r['item_id']:r for r in cached}
    assert len(indexed)==len(cached)
    missing=[r for r in rows if r['item_id'] not in indexed]
    for out in original.generate_pairs(missing) if missing else []:indexed[out['item_id']]=out
    originals=[indexed[r['item_id']] for r in rows]
    outputs=pipeline.repair_pairs(rows,originals)
    if cached:
        assert all((out['english'],out['singlish'])==(old['english'],old['singlish'])
            for out,old in zip(outputs,originals,strict=True) if out['context_policy']['action']=='keep_model')
    report=dict(status='complete',account='hm.teoh.2024',gpu=torch.cuda.get_device_name(0),
        tests_passed=checked.testsRun,cases=len(rows),inputs_sha256=sha(args.inputs),
        policy_source_sha256=sha(ROOT/'sealion_context_guard_v1.py'),
        source_bundle=str(BUNDLE),cached_original_cases=len(cached),new_original_cases=len(missing),
        training_performed=False,default_changed=False,
        scope='Experimental hybrid: original trained SEA-LION plus explicit-context rules.71 development/diagnostic inputs, including10 assistant-authored smoke cases now inspected during policy development; no independent human accuracy claim. Rule English is labelled separately; Singlish is neural or labelled English fallback.',
        rule_counts=dict(Counter(out['context_policy']['rule'] or 'keep_model' for out in outputs)),
        english_changed=sum(out['english']!=out['original_english'] for out in outputs),
        singlish_english_fallbacks=sum(out['singlish_status']=='english_fallback' for out in outputs),
        outputs=[dict(out,context=row['context'],speakers=row['speakers'],message=row['message'],scope=row.get('scope'))
            for row,out in zip(rows,outputs,strict=True)],seconds=time.monotonic()-started)
    report['source_adapters_unchanged']=(sha(BUNDLE/'adapters/english/adapter_model.safetensors')=='55a7f02626c08894bfdc4b44f68dacf86e8f4edafb7aa6bd78aead9162698913'
        and sha(BUNDLE/'adapters/singlish/adapter_model.safetensors')=='6ea7066deda4f4e9e5f4a9e070d46427536a0dac05d5a416864e97c2dc3f5db2')
    assert report['source_adapters_unchanged']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='outputs'}),flush=True)


if __name__=='__main__':main()
