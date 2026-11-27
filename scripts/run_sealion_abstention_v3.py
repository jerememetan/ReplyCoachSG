"""Role-corrected abstention SFT with factual self-contained controls."""
import gc
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_abstention_v3_core as experiment
core=experiment.core
plain_metrics=experiment.abstain.metrics
plain_eligible=experiment.abstain.eligible


def robust_metrics(rows):
    result=plain_metrics(rows)
    layouts=sorted({r['layout'] for r in rows if 'layout' in r})
    if layouts:
        result['layout_metrics']={name:plain_metrics([r for r in rows if r.get('layout')==name]) for name in layouts}
        result['aggregate_balanced_score']=result['answerability_balanced_score']
        result['answerability_balanced_score']=min(x['answerability_balanced_score'] for x in result['layout_metrics'].values() if x['missing_cases'] and x['answerable_cases'])
        result['selection_score_note']='Worst of the two mixed answerability history layouts; answerable-only factual controls gate eligibility separately.128rows are not128 independent history scenarios.'
    return result


def robust_eligible(scores,baseline):
    if 'layout_metrics' not in scores:return plain_eligible(scores,baseline)
    return all(plain_eligible(value,baseline['layout_metrics'][name]) for name,value in scores['layout_metrics'].items())


def main():
    assert Path.home().name=='hm.teoh.2024'
    core.torch.set_num_threads(4)
    experiment.abstain.metrics=robust_metrics
    experiment.abstain.eligible=robust_eligible
    recipe=experiment.read('recipe.json')
    groups={name:experiment.read('data/'+name+'.json') for name in ['train','control','validation','challenge','legacy']}
    tok=core.checkpoint_tokenizer(core.BUNDLE/'base')
    experiment.preflight(tok,recipe,groups)
    diagnostics=json.loads((Path.home()/'ReplyCoachSG_sealion_abstention_v2/custom_check/inputs.json').read_text())
    diagnostics=[dict(r,requires_refusal=None,reference_values=[],scope='reused_user_diagnostic') for r in diagnostics]
    cases=groups['challenge']+random.Random(recipe['seed']).sample(groups['legacy'],20)+diagnostics
    experiment.save('evaluation_cases.json',cases)
    started=time.monotonic()
    model=core.load();model.load_adapter(core.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
    experiment.paired_evaluation(model,tok,cases,'outputs/baseline/evaluation.json')
    core.activate(model,'default')
    validation=experiment.english_evaluation(model,tok,groups['validation'],'outputs/baseline/validation.json')
    del model;gc.collect();core.torch.cuda.empty_cache()
    report=dict(status='training',recipe=recipe,arms={},gpu=core.torch.cuda.get_device_name(0),
        original_baseline_complete=True,reset_from_original_after_corpus_correction=True)
    experiment.save('results.json',report)
    report['arms']['selective_abstention']=experiment.train_arm('selective_abstention',tok,groups,recipe,cases,validation)
    assert core.sha(core.BUNDLE/'adapters/english/adapter_model.safetensors')==recipe['source_english_sha256']
    assert core.sha(core.BUNDLE/'adapters/singlish/adapter_model.safetensors')==recipe['frozen_singlish_sha256']
    report.update(status='complete',seconds=time.monotonic()-started,source_adapters_unchanged=True,
        default_changed=False,human_review_pending=True)
    experiment.save('results.json',report)
    print('COMPLETE corrected v3',report['arms']['selective_abstention']['selected_step'],flush=True)


if __name__=='__main__':main()
