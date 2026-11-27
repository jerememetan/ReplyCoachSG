"""Read-only generation review of the best target-validation layout snapshot."""
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_turn_repair_v1 as study
import run_sealion_turn_layout_v2 as layout


def main():
    assert Path.home().name=='hm.teoh.2024'
    study.ROOT=ROOT/'layout_v2'
    study.torch.set_num_threads(4)
    report=study.read('results.json');assert report['status']=='complete'
    history=report['arms']['supervised']['history']
    entry=min(history[1:],key=lambda row:row['scores']['targeted_positive_mean_ce'])
    step=entry['step'];path=study.ROOT/f'outputs/supervised/step_{step:04d}'
    study.source=layout.marked_source
    study.encode=layout.marked_encode
    study.BOUNDARY=re.compile(r'\n(?:(?:Me|Friend|Other)(?: \([A-Z_]+\))?|Earlier conversation|Latest incoming message from Friend):')
    tok=study.checkpoint_tokenizer(study.BUNDLE/'base')
    model=study.load(path)
    model.load_adapter(study.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
    name='supervised/best_target_validation_exploratory'
    result=study.evaluate(model,tok,study.read('evaluation_cases.json'),name)
    result.update(scope='16 prospective synthetic scenarios frozen before this study,1 reused user diagnostic,20 reused development regressions. Exploratory snapshot rejected by retention guard if specified.',source_format='explicit_latest_me_v2')
    study.save(f'outputs/{name}/evaluation.json',result)
    meta=dict(status='complete',step=step,checkpoint=str(path),sha256=study.sha(path/'adapter_model.safetensors'),
        retention_guard_passed=entry['retention_guard_passed'],default_changed=False,
        rationale='Lowest16-case synthetic target validation positive CE among trained snapshots. Challenge/user outputs not used to choose snapshot.',
        flags=result['flags'],training_performed=False)
    study.save('snapshot_review.json',meta)
    print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
