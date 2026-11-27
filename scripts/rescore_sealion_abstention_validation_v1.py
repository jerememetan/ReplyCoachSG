"""Recompute generated-validation scores after a documented preference-label audit."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import sealion_abstention_v1 as abstain


def read(name):return json.loads((ROOT/name).read_text())


def main():
    assert Path.home().name=='hm.teoh.2024'
    assert read('results.json')['status']=='complete'
    record=read('validation_amendment/record.json')
    assert hashlib.sha256((ROOT/'validation_amendment/validation.json').read_bytes()).hexdigest()==record['corrected_validation_sha256']
    expected={r['item_id']:r for r in read('validation_amendment/validation.json')}
    report=dict(status='complete',amendment=record,training_performed=False,model_inference_performed=False,arms={})
    for arm in ['restored_context_control','selective_abstention']:
        rows=[]
        for step in [0,70,140,280]:
            original=read(f'outputs/{arm}/validation_{step:04d}.json')
            corrected=[]
            for old in original['outputs']:
                spec=expected[old['item_id']]
                assert old['context']==spec['context'] and old['message']==spec['message']
                corrected.append(dict(old,requires_refusal=spec['requires_refusal'],
                    reference_values=spec['reference_values'],positive=spec['positive']))
            scores=abstain.metrics(corrected)
            rows.append(dict(step=step,metrics=scores))
        initial=rows[0]['metrics'];best=initial['answerability_balanced_score'];chosen=0
        for row in rows[1:]:
            row['eligible']=abstain.eligible(row['metrics'],initial)
            row['selected']=row['eligible'] and row['metrics']['answerability_balanced_score']>best
            if row['selected']:best=row['metrics']['answerability_balanced_score'];chosen=row['step']
        original_step=read(f'outputs/{arm}/training.json')['selected_step']
        checkpoint=str(Path.home()/'ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1/adapters/english') if chosen==0 else str(ROOT/f'outputs/{arm}/step_{chosen:04d}')
        report['arms'][arm]=dict(original_selected_step=original_step,corrected_selected_step=chosen,
            selected_step_changed=chosen!=original_step,checkpoint=checkpoint,history=rows)
    path=ROOT/'validation_amendment/results.json'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
