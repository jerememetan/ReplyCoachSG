"""Freeze the four-arm, matched-primary-exposure mT5 multi-task pilot."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from transformers import AutoTokenizer
from replycoach.generate.context_pair_ablation_v1 import main_rows, encode_row
from replycoach.generate.mt5_cocktail_v1 import make_cases, training_schedule, auxiliary_source, ARMS

OUT=ROOT/'artifacts/mt5_cocktail_v1'
TRAIN='data/training_expansion/generated600_repair_v1/combined_train.jsonl'
TEAM_SHA='03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def rows(p):return [json.loads(x) for x in (ROOT/p).read_text().splitlines() if x.strip()]
def frozen(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if p.exists() and p.read_text()!=text:raise FileExistsError(p)
    p.write_text(text);return p


def main():
    team=ROOT/'data/dev/train.jsonl';expanded=ROOT/TRAIN
    assert sha(team)==TEAM_SHA and expanded.read_bytes().startswith(team.read_bytes())
    train,valid=rows(TRAIN),rows('data/dev/validation.jsonl')
    audit=read('data/training_expansion/sealion_rewrite_v1/pair_audit.json')
    assert sha(expanded)==audit['source_sha256']
    eligible=lambda split:{r['id'] for r in audit['items'] if r['split']==split and r['eligible']}
    primary=main_rows(train,eligible('train'));val=main_rows(valid,eligible('validation'))
    assert len(train)==790 and len(primary)==1535 and len(val)==92
    assert not {r['id'] for r in train}&{r['id'] for r in valid}
    aux={s:make_cases(s,n) for s,n in [('train',32),('validation',6),('challenge',4)]}
    key=lambda r:json.dumps([r['context'],r['speakers'],r['message']],sort_keys=True)
    groups={'primary':primary,'validation':val,**aux}
    for a,b in [('train','validation'),('train','challenge'),('validation','challenge')]:
        assert not {key(r) for r in aux[a]}&{key(r) for r in aux[b]}
    assert not {key(r) for r in primary}&{key(r) for r in aux['validation']+aux['challenge']}
    legacy=[dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],scope='reused_project_validation') for r in valid]
    diagnostics=[dict(item_id='USER-CALL-001',context=['Can u call me now?','Eh i not so free eh?','Ohh, then when can we call?'],speakers=['user','peer','user'],message='I am not sure, what do you want to call me about',scope='reused_user_diagnostic'),
        dict(item_id='CALL-WITH-PURPOSE',context=['I wanted to ask you about our assignment deadline. Can you call me now?','I am not free right now.','Okay, we can speak later.'],speakers=['user','peer','user'],message='What did you want to call me about?',scope='reused_user_diagnostic')]
    natural=[]
    scenarios=[
        (['I cannot do Monday; Wednesday is my free day.','Let me check the group schedule.'],['user','peer'],'Shall we rehearse on Monday at six?','Do not accept unavailable Monday; Wednesday may be proposed.'),
        (['Wednesday is packed for me, but Monday is free.','Let me check the group schedule.'],['user','peer'],'Shall we rehearse on Monday at six?','Monday is available; do not claim it is the busy day.'),
        (['My laptop charger has stopped working.','Mine is in my bag.'],['peer','user'],'Could I borrow yours until tomorrow?','Friend asks to borrow User charger; reply must preserve lending direction.'),
        (['My laptop charger has stopped working.','Mine is in my bag.'],['user','peer'],'Want to borrow it until tomorrow?','Friend offers their charger; User would be borrowing, not lending.'),
        (['I have eight dollars left for lunch.','The vegetarian bowl here costs sixteen dollars.'],['user','peer'],'Want to get that bowl together?','Do not claim sixteen-dollar bowl fits eight-dollar budget.'),
        (['I have twenty dollars left for lunch.','The vegetarian bowl here costs sixteen dollars.'],['user','peer'],'Want to get that bowl together?','Sixteen-dollar bowl is within twenty-dollar budget; preferences remain open.'),
        (['I emailed you the completed slides this morning.','I cannot find them in my inbox.'],['user','peer'],'Could you send the slides again now?','A resend or checking address is appropriate; do not claim slides were never completed.'),
        (['I have not finished making the slides yet.','The presentation is tomorrow morning.'],['user','peer'],'Could you send the completed slides now?','Do not falsely claim completed slides are ready or already sent.'),
        (['I wanted to talk about fixing the brake cable on my bicycle.','I am in a tutorial right now.','No rush, we can talk after class.'],['user','peer','user'],'What was the call going to be about?','Purpose is bicycle brake cable repair; preserve that topic.'),
        (['Do you have a moment for a call?','I am in a tutorial right now.','No rush, we can talk after class.'],['user','peer','user'],'What was the call going to be about?','No purpose supplied; do not invent a private topic. Flag missing information or give a neutral clarification.'),
        (['I can only leave the lab after four.','The cafe closes at six.'],['user','peer'],'Could we meet at half past three?','Do not accept 15:30 given earliest availability after16:00.'),
        (['I can leave the lab after three.','The cafe closes at six.'],['user','peer'],'Could we meet at half past three?','15:30 is compatible with after15:00; do not claim it is too early.'),
    ]
    for i,(context,speakers,message,constraints) in enumerate(scenarios):
        natural.append(dict(item_id=f'COCKTAIL-NATURAL-{i+1:02d}',pair_id=i//2,context=context,speakers=speakers,message=message,
            scope='fresh_assistant_natural_challenge',review_constraints=constraints))
    assert not {key(r) for r in primary+aux['train']} & {key(r) for r in diagnostics}
    assert not {key(r) for r in primary+aux['train']+aux['validation']+legacy} & {key(r) for r in natural}
    tok=AutoTokenizer.from_pretrained('/tmp/replycoach_mt5_adapted_tokenizer',local_files_only=True)
    budgets={}
    for label,records in [('primary',primary),('project_validation',val)]:
        encoded=[encode_row(r,tok) for r in records]
        budgets[label]=dict(count=len(records),max_source=max(len(x) for x,y in encoded),max_target=max(len(y) for x,y in encoded))
    for split,records in aux.items():
        for task,field in [('FACTS','fact_target'),('ANSWERABILITY','answer_target')]:
            enc=[(tok(auxiliary_source(r,task))['input_ids'],tok(r[field])['input_ids']) for r in records]
            assert all(len(x)<=192 and len(y)<=48 and y[-1]==tok.eos_token_id for x,y in enc)
            budgets[split+'_'+task]=dict(count=len(enc),max_source=max(len(x) for x,y in enc),max_target=max(len(y) for x,y in enc))
    mapping={
        'data/primary.json':frozen(OUT/'data/primary.json',primary),
        'data/project_validation.json':frozen(OUT/'data/project_validation.json',val),
        'data/legacy.json':frozen(OUT/'data/legacy.json',legacy),
        'data/diagnostics.json':frozen(OUT/'data/diagnostics.json',diagnostics),
        'data/natural_challenge.json':frozen(OUT/'data/natural_challenge.json',natural),
        'schedule.json':frozen(OUT/'schedule.json',training_schedule(len(primary),len(aux['train']))),
        'source_manifest.json':ROOT/'experiments/mt5_restored_pipeline_v1_source_manifest.json',
    }
    for split,records in aux.items():mapping['data/'+split+'.json']=frozen(OUT/'data'/f'{split}.json',records)
    files=list(ROOT.glob('src/**/*.py'))+[ROOT/p for p in [
        'scripts/run_mt5_cocktail_v1.py','scripts/prepare_mt5_cocktail_v1.py',
        'scripts/run_mt5_adapter_switch_v1.py','tests/test_mt5_cocktail_v1.py',
        'data/dev/train.jsonl',TRAIN,'data/dev/validation.jsonl',
        'data/training_expansion/sealion_rewrite_v1/pair_audit.json']]
    mapping.update({str(p.relative_to(ROOT)):p for p in files})
    recipe=dict(experiment='mt5_cocktail_v1',account='hm.teoh.2024',seed=1,
        research_source='https://arxiv.org/html/2410.01109v1',
        hypothesis='Auxiliary speaker/fact extraction and answerability supervision improve contextual English replies while preserving Singlish meaning.',
        topology='Same fresh shared rank16q/v LoRA(alpha32,dropout0.1) on the preserved582401280parameter publicly adapted mT5-base;1769472trainable parameters.',
        source_base_personal='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_mt5_mlp_generation_v1/model',
        arms={a:dict(facts_weight=f,answerability_weight=b) for a,(f,b) in ARMS.items()},
        main_pool=dict(english=790,rewrite=745,unique_conversations=790),
        auxiliary_pool=dict(train_conversations=800,validation_conversations=160,challenge_conversations=104,
            tasks=['FACTS','ANSWERABILITY'],provenance='Deterministic assistant-authored templates with an audited speaker/evidence ledger; not human gold or API output.'),
        schedule=dict(optimizer_steps=784,primary_per_step=12,facts_per_step=2,answerability_per_step=2,
            primary_exposures=9408,facts_forward_exposures=1568,answerability_forward_exposures=1568),
        objective='Token-mean primary CE + facts_weight*token-mean fact CE + answerability_weight*token-mean answerability CE. All arms forward/backpropagate the same auxiliary batches, with zero coefficients for inactive tasks.',
        optimizer=dict(name='AdamW',learning_rate=3e-4,weight_decay=.01,clip_norm=1.,precision='FP32'),
        snapshots=[0,128,256,512,784],
        selection='Minimum mean English/rewrite project-validation token CE including step0; ties retain earlier. Aux accuracy and all challenge/user outputs excluded from selection. Final784 also reported.',
        inference=dict(num_beams=4,no_repeat_ngram_size=3,do_sample=False,max_new_tokens=48,max_source=192,
            primary='Generate English then rewrite that exact generated anchor with the SAME shared adapter. Aux task predictions reported separately, never used as oracle facts or forced reply gates.'),
        matching='Identical initialization, primary and auxiliary schedules, shapes, per-step dropout seeds, optimizer and update count. Active task coefficients differ. Extra nonzero auxiliary gradients are the intervention. Equal forwards do not imply identical wall time or equal useful supervised tokens.',
        paper_deviations='A focused four-arm LoRA pilot on mT5 rather than full fine-tuning of instruction-tuned decoder models; weighted fixed-composition batches rather than a uniform concatenation. No general-instruction/math dataset sweep. One seed only.',
        evaluation=dict(project_validation_reused=50,fresh_synthetic=104,fresh_underlying_fact_groups=32,self_contained_controls=8,fresh_natural_challenge=12,natural_contrast_pairs=6,reused_user_diagnostics=2,
            note='104synthetic rows comprise32 groups with supported/other-owner/irrelevant variants plus8controls. Aux exact match is not reply accuracy. Review roles, grounded facts, meaning preservation and naturalness separately; no independent human/locked test score.'),
        budgets=budgets,team_original_sha256=TEAM_SHA,original790_unchanged=True,
        locked_test_read=False,default_changed=False,
        files_sha256={name:sha(path) for name,path in mapping.items()})
    frozen(OUT/'recipe.json',recipe);frozen(ROOT/'experiments/mt5_cocktail_v1_recipe.json',recipe)
    with tarfile.open(OUT/'payload.tgz','w:gz') as tar:
        for name,path in mapping.items():tar.add(path,arcname=name,recursive=False)
        tar.add(OUT/'recipe.json',arcname='recipe.json',recursive=False)
    print(json.dumps(dict(primary=len(primary),auxiliary={k:len(v) for k,v in aux.items()},budgets=budgets,payload_sha256=sha(OUT/'payload.tgz'))))


if __name__=='__main__':main()
