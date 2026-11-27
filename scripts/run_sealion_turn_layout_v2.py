"""Train an explicit latest-turn/Me serialization while preserving the old bundle."""
import gc
from pathlib import Path
import random
import re
import sys
import time

PARENT=Path(__file__).resolve().parent
sys.path.insert(0,str(PARENT))
import run_sealion_turn_repair_v1 as study


def marked_source(row,style='STANDARD'):
    assert style=='STANDARD'
    speakers={'user':'Me','peer':'Friend','other':'Other'}
    turns=[speakers[s]+': '+text for s,text in zip(row['speakers'],row['context'],strict=True)]
    return '\n'.join(['Earlier conversation:',*turns,
        'Latest incoming message from Friend:',row['message'],'Me (STANDARD):'])


def marked_encode(row,tok,key='positive'):
    prompt=tok.encode(marked_source(row),add_special_tokens=False)
    reply=tok.encode(row[key],add_special_tokens=False)+[tok.eos_token_id]
    assert prompt and row[key].strip() and len(prompt)+len(reply)<=384
    return dict(input_ids=prompt+reply,attention_mask=[1]*(len(prompt)+len(reply)),
        labels=[-100]*len(prompt)+reply,prompt_tokens=len(prompt))


def main():
    assert Path.home().name=='hm.teoh.2024'
    study.ROOT=PARENT/'layout_v2'
    study.torch.set_num_threads(4)
    recipe=study.read('recipe.json')
    groups={name:study.read('data/'+name+'.json') for name in ['train','validation','challenge','ordinary','legacy','diagnostic']}
    tok=study.checkpoint_tokenizer(study.BUNDLE/'base')
    regression=random.Random(recipe['seed']).sample(groups['legacy'],20)
    cases=[dict(r,scope='prospective_synthetic_challenge') for r in groups['challenge']]
    cases += [dict(r,scope='user_diagnostic') for r in groups['diagnostic']]
    cases += [dict(r,scope='reused_development_regression') for r in regression]
    study.save('evaluation_cases.json',cases)
    evaluator=study.evaluate
    source_format='original_canonical'
    def evaluate(model,tok,cases,name,adapter='default',expanded=False):
        result=evaluator(model,tok,cases,name,adapter,expanded)
        result['scope']='16 new prospective synthetic scenarios,1 reused user diagnostic,20 reused development regressions. No human/test accuracy claim.'
        result['source_format']=source_format
        study.save(f'outputs/{name}/evaluation.json',result)
        return result
    study.evaluate=evaluate
    model=study.load();model.load_adapter(study.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
    original=study.evaluate(model,tok,cases,'original_canonical')
    assert not original['flags']
    study.source=marked_source
    study.encode=marked_encode
    study.BOUNDARY=re.compile(r'\n(?:(?:Me|Friend|Other)(?: \([A-Z_]+\))?|Earlier conversation|Latest incoming message from Friend):')
    source_format='explicit_latest_me_v2'
    study.preflight(tok,recipe,groups)
    untrained=study.evaluate(model,tok,cases,'untrained_layout')
    del model;gc.collect();study.torch.cuda.empty_cache()
    started=time.monotonic()
    report=dict(status='training',recipe=recipe,original_baseline_complete=True,
        untrained_layout_complete=True,arms={},gpu=study.torch.cuda.get_device_name(0))
    study.save('results.json',report)
    report['arms']['supervised']=study.train_arm('supervised',tok,groups,recipe,cases)
    assert study.sha(study.BUNDLE/'adapters/english/adapter_model.safetensors')==recipe['initial_english_sha256']
    assert study.sha(study.BUNDLE/'adapters/singlish/adapter_model.safetensors')==recipe['frozen_singlish_sha256']
    report.update(status='complete',seconds=time.monotonic()-started,
        source_adapters_unchanged=True,default_changed=False,human_review_pending=True)
    study.save('results.json',report)
    print('COMPLETE layout_v2 selected',report['arms']['supervised']['selected_step'],flush=True)


if __name__=='__main__':main()
