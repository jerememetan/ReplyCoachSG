"""Read-only best-target-validation snapshot and input-sensitivity diagnostics."""
import gc
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_turn_repair_v1 as study


def main():
    assert Path.home().name=='hm.teoh.2024'
    study.torch.set_num_threads(4)
    report=study.read('results.json');assert report['status']=='complete'
    cases=study.read('evaluation_cases.json')
    tok=study.checkpoint_tokenizer(study.BUNDLE/'base')
    reviewed={}
    for arm in ['supervised','contrastive']:
        training=study.read(f'outputs/{arm}/training.json')
        entry=min(training['history'][1:],key=lambda row:row['scores']['targeted_positive_mean_ce'])
        step=entry['step'];path=ROOT/f'outputs/{arm}/step_{step:04d}'
        model=study.load(path)
        model.load_adapter(study.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
        result=study.evaluate(model,tok,cases,f'{arm}/best_target_validation_exploratory')
        reviewed[arm]=dict(step=step,selected_by_retention_guard=False,
            rationale='Lowest synthetic target validation positive CE among trained snapshots, irrespective of old-reference CE retention guard. No challenge/user output used for this choice.',
            checkpoint=str(path),sha256=study.sha(path/'adapter_model.safetensors'),flags=result['flags'])
        del model;gc.collect();study.torch.cuda.empty_cache()
    original=study.read('data/diagnostic.json')[0]
    variants=[]
    def add(ident,message,context=None,note=''):
        variants.append(dict(original,item_id='CALL-FORMAT-'+ident,message=message,
            context=original['context'] if context is None else context,
            scope='input_sensitivity_diagnostic',note=note))
    add('exact',original['message'],note='Exact user input; no changes.')
    add('question_mark',original['message']+'?',note='Only append missing final question mark.')
    add('spell_you',original['message'],[x.replace(' u ',' you ') for x in original['context']],note='Only expand u to you in first context turn.')
    add('question_only','What do you want to call me about?',note='Remove uncertainty preface and add question mark; purpose question retained.')
    add('formal','I am not sure. What would you like to discuss on the call?',
        ['Could you call me now?','I am not free right now.','Okay, when could we talk?'],
        note='Semantically corresponding formal wording; not exact original input.')
    add('mixed_informal','Not sure leh, but what you want to talk about?',note='Informal two-clause latest incoming paraphrase.')
    add('known_purpose',original['message'],
        ['I want your advice about my internship application. Can you call me now?',*original['context'][1:]],
        note='Adds a known purpose as a controlled diagnostic. Not the user original context.')
    add('current_question','What is the reason you asked me to call you?',note='Explicitly resolves who initiated the call.')
    study.save('input_sensitivity_cases.json',variants)
    model=study.load();model.load_adapter(study.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
    study.evaluate(model,tok,variants,'input_sensitivity_baseline')
    study.save('snapshot_review.json',dict(status='complete',reviews=reviewed,
        scope='Exploratory generation review. Original validation guard remains unchanged. Formatting variants are diagnostics, not eight unseen independent cases.',
        default_changed=False,training_performed=False))
    print(json.dumps(reviewed),flush=True)


if __name__=='__main__':main()
