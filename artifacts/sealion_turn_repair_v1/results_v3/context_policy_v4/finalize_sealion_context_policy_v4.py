"""CPU-only extra rewrite check on actual cached job358077 generations."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parent
BUNDLE=Path.home()/'ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1'
sys.path[:0]=[str(ROOT),str(BUNDLE/'src'),str(BUNDLE/'deps')]
import sealion_context_guard_v1 as policy
import sealion_context_pipeline_v1 as guarded


def main():
    assert Path.home().name=='hm.teoh.2024'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=json.loads((ROOT/'payload_manifest.json').read_text())
    assert all(sha(ROOT/name)==value for name,value in manifest.items())
    import replycoach.generate
    sys.modules['replycoach.generate.sealion_context_guard_v1']=policy
    sys.modules['replycoach.generate.sealion_context_pipeline_v1']=guarded
    checked=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_sealion_context_guard_v1.py'))
    assert checked.wasSuccessful() and checked.testsRun==31
    source=ROOT.parent/'context_policy_v3/results.json'
    report=copy.deepcopy(json.loads(source.read_text()))
    assert report['status']=='complete' and len(report['outputs'])==71
    changes=[]
    for row in report['outputs']:
        if row['context_policy']['action']=='keep_model' or row['singlish_status']=='english_fallback':continue
        check=guarded.check_rewrite(row['english'],row['raw_repaired_singlish'])
        row['rewrite_guard']=check
        if not check['passed']:
            changes.append(row['item_id'])
            row.update(singlish=row['english'],singlish_status='english_fallback',
                singlish_provenance='english_fallback_rewrite_guard')
    assert sha(BUNDLE/'adapters/english/adapter_model.safetensors')=='55a7f02626c08894bfdc4b44f68dacf86e8f4edafb7aa6bd78aead9162698913'
    assert sha(BUNDLE/'adapters/singlish/adapter_model.safetensors')=='6ea7066deda4f4e9e5f4a9e070d46427536a0dac05d5a416864e97c2dc3f5db2'
    report.update(tests_passed=31,policy_revision='v4_additional_past_conversation_check',
        model_inference_performed=False,postprocessing_performed=True,source_inference_job=358077,
        source_results_sha256=sha(source),additional_english_fallback_ids=changes,
        singlish_english_fallbacks=sum(row['singlish_status']=='english_fallback' for row in report['outputs']),
        final_pipeline_source_sha256=sha(ROOT/'sealion_context_pipeline_v1.py'))
    assert not (ROOT/'results.json').exists()
    (ROOT/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='complete',tests=31,postprocessing_only=True,changes=changes,
        fallbacks=report['singlish_english_fallbacks'])),flush=True)


if __name__=='__main__':main()
