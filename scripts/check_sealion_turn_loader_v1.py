"""Check the original export and study loader on one GPU and exact user input."""
import gc
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_turn_repair_v1 as study
from peft import get_peft_model_state_dict


def main():
    assert Path.home().name=='hm.teoh.2024'
    study.torch.set_num_threads(4)
    rows=study.read('data/diagnostic.json')
    spec=importlib.util.spec_from_file_location('official_export',study.BUNDLE/'infer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pipeline=module.ReplyPipeline(study.BUNDLE,device='cuda',verify=False)
    official=pipeline.generate_pairs(rows)
    state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(pipeline.model,adapter_name='english').items()}
    prompt_ids=pipeline.tokenizer.encode(study.source(rows[0],'STANDARD'),add_special_tokens=False)
    del pipeline;gc.collect();study.torch.cuda.empty_cache()
    model=study.load();model.load_adapter(study.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
    other=get_peft_model_state_dict(model,adapter_name='default')
    assert set(state)==set(other)
    equal=all(study.torch.equal(v,other[k].detach().cpu()) for k,v in state.items())
    assert equal
    tok=study.checkpoint_tokenizer(study.BUNDLE/'base')
    assert tok.encode(study.source(rows[0],'STANDARD'),add_special_tokens=False)==prompt_ids
    report=study.evaluate(model,tok,[dict(r,scope='user_diagnostic') for r in rows],'loader_audit_a40')
    outputs=report['outputs']
    differences=[r['item_id'] for r,s in zip(official,outputs,strict=True) if any(r[k]!=s[k] for k in ['english','singlish'])]
    summary=dict(status='complete',gpu=study.torch.cuda.get_device_name(0),adapter_tensors_identical=equal,
        prompt_tokens_identical=True,official=official,study=outputs,differences=differences,
        note='Same GPU original export vs new study loader. Training study uses one L40S for all arms; original one-case demo used A40.')
    study.save('loader_audit.json',summary)
    print(json.dumps(summary),flush=True)
    assert not differences


if __name__=='__main__':main()
