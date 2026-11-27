"""Exact new user conversation on original and three creative SEA-LION variants."""
import csv
import gc
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
STUDY=Path.home()/'ReplyCoachSG_sealion_creative_v1'
sys.path.insert(0,str(STUDY))
import run_sealion_creative_v1 as experiment
core=experiment.core


def main():
    assert Path.home().name=='hm.teoh.2024'
    core.torch.set_num_threads(4)
    assert not (ROOT/'comparison.json').exists()
    inputs=json.loads((ROOT/'inputs.json').read_text())
    spec=importlib.util.spec_from_file_location('original_export',core.BUNDLE/'infer.py')
    export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
    export.verify_files(core.BUNDLE);export.validate_inputs(inputs)
    inventory=json.loads((STUDY/'checkpoint_inventory.json').read_text())
    checkpoints=[dict(label='original_repair25',path=core.BUNDLE/'adapters/english',sha256='55a7f02626c08894bfdc4b44f68dacf86e8f4edafb7aa6bd78aead9162698913')]
    for arm in ['single_reference','multiple_references','multiple_plus_contrast']:
        entry=next(r for r in inventory if r['arm']==arm and r['step']==300)
        checkpoints.append(dict(label=arm+'_final300',path=Path(entry['path']),sha256=entry['sha256']))
    outputs={}
    for item in checkpoints:
        assert core.sha(item['path']/'adapter_model.safetensors')==item['sha256']
        model=core.load(item['path']);model.load_adapter(core.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
        result=experiment.evaluate(model,core.checkpoint_tokenizer(core.BUNDLE/'base'),inputs,ROOT/'outputs'/item['label']/ 'results.json')
        assert core.sha(item['path']/'adapter_model.safetensors')==item['sha256']
        outputs[item['label']]=dict(checkpoint=str(item['path']),sha256=item['sha256'],**result)
        print(item['label'],json.dumps(result['outputs'],ensure_ascii=False),flush=True)
        del model;gc.collect();core.torch.cuda.empty_cache()
    assert core.sha(core.BUNDLE/'adapters/english/adapter_model.safetensors')==checkpoints[0]['sha256']
    assert core.sha(core.BUNDLE/'adapters/singlish/adapter_model.safetensors')=='6ea7066deda4f4e9e5f4a9e070d46427536a0dac05d5a416864e97c2dc3f5db2'
    report=dict(status='complete',inputs=inputs,inputs_sha256=core.sha(ROOT/'inputs.json'),outputs=outputs,
        gpu=core.torch.cuda.get_device_name(0),training_performed=False,source_weights_unchanged=True,
        default_changed=False,original_backup_modified=False,
        decoding='Primary beam4/no-repeat3/max64; supplemental3seeded nucleus samples attemperature0.8/top_p0.9, unranked. Frozen original Singlish rewrites each exact generated English draft.',
        note='Exact user text/roles; no manually supplied timing, target reply, oracle context, refusal gate, rule layer or reranker.')
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    with (ROOT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','english','singlish','english_status','singlish_status']);writer.writeheader()
        for name,r in outputs.items():
            out=r['outputs'][0];writer.writerow(dict(model=name,**{k:out[k] for k in ['english','singlish','english_status','singlish_status']}))
    print('CUSTOM MEETING INFERENCE COMPLETE',flush=True)


if __name__=='__main__':main()
