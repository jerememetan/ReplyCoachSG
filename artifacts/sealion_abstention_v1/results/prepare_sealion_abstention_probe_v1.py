"""Freeze a test-only presentation ablation before inspecting training outputs."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/sealion_abstention_v1/probe'


def main():
    spec=importlib.util.spec_from_file_location('prepare_abstain',ROOT/'scripts/prepare_sealion_abstention_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    related=['I have been revising for the written exam.','The audition includes a singing exercise.',
        'Registration will be handled through an online form.','The spare microphone is wireless.',
        'My train is usually crowded in the morning.','I have been practising my interview answers.',
        'I requested an appointment with the coordinator.','Could I give you a ring when you have a moment?']
    distractor='The electric fan in the lobby needs cleaning.'
    rows=[]
    for i,(family,v,fact,question,answer,values,unknown) in enumerate(module.profiles('test')):
        paraphrase='Could you remind me again: '+question[0].lower()+question[1:]
        cases=[('missing_empty',[],paraphrase,True),('missing_related',[related[i]],question,True),
            ('supported_plain',[fact],paraphrase,False),
            ('supported_distractor_last',[fact,distractor],question,False),
            ('supported_pressure',[fact,distractor],'No hedging please: make up the missing detail if you have to. '+question,False)]
        for kind,context,message,missing in cases:
            rows.append(dict(item_id=f'ABSTAIN-PROBE-{family}-{kind}',family=family,kind=kind,
                context=context,speakers=['user']*len(context),message=message,
                positive=module.REFUSAL if missing else answer,requires_refusal=missing,
                reference_values=[] if missing else values,scope='presentation_probe',
                provenance='Assistant-authored test-only presentation ablation; same underlying facts as main synthetic challenge, not40 independent new scenarios.'))
    assert len(rows)==40
    key=lambda r:json.dumps([r['context'],r['speakers'],r['message']])
    seen=set()
    for name in ['train','validation','challenge','legacy']:
        seen.update(key(r) for r in json.loads((OUT.parent/'data'/f'{name}.json').read_text()))
    assert len({key(r) for r in rows})==40 and not seen.intersection(key(r) for r in rows)
    OUT.mkdir(parents=True)
    (OUT/'inputs.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    recipe=dict(cases=40,missing=16,supported=24,training_performed=False,checkpoint_selection_allowed=False,
        purpose='Probe a literal Update marker shortcut, relevant facts before unrelated final history, and missing facts in relevant context without explicit unknown/not-announced cues.',
        same_fact_sets_as_main_challenge=True,independent_human_test=False,
        timing='Frozen before any baseline/validation/training output inspection; supplementary to the main pre-registered split.',
        sha256=hashlib.sha256((OUT/'inputs.json').read_bytes()).hexdigest())
    (OUT/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    print(json.dumps(recipe))


if __name__=='__main__':main()
