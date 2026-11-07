"""Freeze a bounded public-adaptation -> paired-project comparison."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/evaluation/sealion_dialogue_v1'
PUBLIC=ROOT/'data/public/sealion_dialogue_v1'
EXAMPLES=[
 (['I cannot leave work before8:10.','We can get dinner afterwards.'],['user','peer'],'7:40 can meet already or not?','Do not agree to meet before8:10 without resolving constraint.'),
 (['You still have my headphones.','Yes, I kept them in their case.'],['user','peer'],'Can I return your headphones tomorrow afternoon?','PEER holds USER headphones; no owner/returner reversal.'),
 (['I have not checked the community room closing hours.','They may change during holidays.'],['user','peer'],'Does the room close at ten tonight?','Hours explicitly unknown; uncertainty/checking/deferral valid.'),
 (['I could not attend the briefing.','I took down the instructions.'],['user','peer'],'Want me to send you my briefing notes?','PEER offers notes to USER; respond as recipient.'),
 (['I have the event photos saved on my laptop.','I missed the event.'],['user','peer'],'Can you send me the photos later?','USER sends photos to PEER; no direction reversal.'),
 (['I only have six dollars for a snack today.','There are stalls near the station.'],['user','peer'],'Want to get the fifteen-dollar pastry set?','Respect six-dollar budget; valid alternatives/refusals allowed.'),
 (['I uploaded the signed form last night.','I will check the folder.'],['user','peer'],'Have you uploaded the form yet?','Already uploaded; do not claim it is still undone.'),
 (['I have not started making the slides.','The meeting is later this week.'],['user','peer'],'Your slides ready already?','Not started; do not claim completed.'),
 (['I failed the assessment.','Oh no, how are you feeling?'],['peer','user'],'Quite disappointed, I studied hard for it.','Support PEER disappointment, no success congratulations.'),
 (['I am free on Sunday afternoon.','We are planning a small picnic.'],['user','peer'],'Want to join our picnic?','Any sensible acceptance/refusal/clarification valid; no exact preferred act.'),
 (['I am waiting outside room C2.','I am on my way.'],['user','peer'],'Which room are you outside?','Respect known outside room C2, not inside or another room.'),
 (['I do not eat meat.','We can choose something nearby.'],['user','peer'],'Shall I order the chicken rice for you?','Do not accept meat as compatible without resolving dietary constraint.'),
]


def signature(r):return json.dumps([r['context'],r['speakers'],r['message']],separators=(',',':')).lower()


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    inputs=[];notes=[]
    for i,(context,speakers,message,note) in enumerate(EXAMPLES,1):
        key=f'SEAPUBLIC-{i:03d}'
        context=[s.replace('before8:10','before 8:10') for s in context]
        inputs.append(dict(item_id=key,context=context,speakers=speakers,message=message))
        notes.append(dict(item_id=key,expectation=note,provenance='assistant-authored development guidance, not independent human gold'))
    (FOLDER/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (FOLDER/'review_guidance.json').write_text(json.dumps(notes,indent=2)+'\n')
    manifest=json.loads((PUBLIC/'manifest.json').read_text())
    if manifest['status']!='ready_exploratory':raise ValueError('Public sample inspection incomplete')
    old=json.loads((ROOT/'data/evaluation/sealion_dual_v1/inputs.json').read_text())
    train=[json.loads(l) for l in (PUBLIC/'train.jsonl').read_text().splitlines()]
    known={signature(r) for r in train}|{signature(r) for r in old}
    for split in ['train','validation']:
        for line in (ROOT/f'data/dev/{split}.jsonl').read_text().splitlines():
            r=json.loads(line);known.add(signature(dict(context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'])))
    if {signature(r) for r in inputs}&known:raise ValueError('Exact input overlap with new evaluation')
    vendor=json.loads((ROOT/'src/replycoach/vendor/sealion_v1/provenance.json').read_text())
    files=[*sorted(ROOT.glob('src/**/*.py')),
        *[p for p in sorted((ROOT/'src/replycoach/vendor/sealion_v1').iterdir()) if p.is_file()],
        *[ROOT/'scripts'/n for n in ['prepare_sealion_dialogue_v1.py','prepare_sealion_public_data_v1.py','review_sealion_public_pilot_v1.py','run_sealion_dual_v1.py','run_sealion_dialogue_v1.py','v5_sealion_dialogue_v1.sbatch']],
        *[ROOT/'tests'/n for n in ['test_sealion_public_v1.py','test_sealion_dual_v1.py','test_sealion_data.py','test_sealion_v1_compat.py']],
        ROOT/'data/dev/train.jsonl',ROOT/'data/dev/validation.jsonl',
        ROOT/'data/evaluation/sealion_dual_v1/inputs.json',*sorted(FOLDER.glob('*.json')),
        *[p for p in sorted(PUBLIC.iterdir()) if p.is_file()]]
    files=list(dict.fromkeys(files))
    recipe=dict(experiment='sealion_dialogue_v1',repo_id=vendor['repo_id'],revision=vendor['revision'],weights_sha256=vendor['weights']['lfs']['sha256'],
        stages=['bounded public next-reply adaptation under common DIALOGUE tag','original190 conversation xSTANDARD/LOCAL task training','paired automatic usefulness/style comparison'],
        public=dict(training_pairs=7994,validation_pairs=400,train_by_source={k:v['selected_train'] for k,v in manifest['corpora'].items()},max_passes=1,learning_rate=5e-5,microbatch=1,accumulation=16,validation_every_examples=2000,selection='Minimum equal-source public validation CE among0/2000/4000/6000/7994 examples; include zero-adaptation baseline. One bounded pass, no validation gradients.',
            style='DIALOGUE training-only tag for both sources; corpus identity not equated with STANDARD/LOCAL.',cleaning='Existing mT5-relatedness-filtered CoSEM source plus deterministic pattern/window/length/repetition filtering, whole chat/dialogue separation, exact heldout-pair exclusion, coverage strata and40-example assistant inspection with6 exclusions. Not semantic/human gold.'),
        task=dict(train_conversations=190,train_targets=380,validation_conversations=50,validation_targets=100,learning_rate=1e-4,accumulation=8,max_epochs=12,patience=2,selection='Minimum equal-style validation CE among trained task epochs1+; original targets copied exactly, no synthetic/revised replies.',matching='Same original data/optimizer/task order/style prefixes as project-only run; reset optimizer and restore post-adapter-construction RNG state before task phase.'),
        lora=dict(r=8,alpha=16,dropout=.05,target_modules=['Wqkv','out_proj','up_proj','down_proj']),seed=425,precision='BF16 frozen base, FP32 reply-only CE',sequence_limit=384,
        evaluation=dict(new_development_conversations=12,reused_development_conversations=20,reused_project_validation=50,project_only_comparator='Existing exact SEA-LION dual-style best epoch1 adapter from355889, no new control training. Generate both styles on new12 and retain its old20 outputs.',criteria='Any context-compatible useful reply; STANDARD/LOCAL meaning consistency; natural conversational Singlish and readable English. No intent-label/exact-reference requirement or particle-count reward.',scope='New12 assistant-authored structurally related development cases frozen before training outputs; not independent human gold or locked product test. Old20/50 heavily reused. Public adaptation adds data and compute, not an isolated compute-matched causal ablation.',stop='No default promotion or further runs automatically after results.'),
        data_limits=manifest['limitations'],default_changed=False,files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    path=ROOT/'experiments/sealion_dialogue_v1_recipe.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
    # Public text remains in ignored local files/private cluster snapshot, never
    # tracked source. A metadata-only copy is kept with experiment records.
    (ROOT/'experiments/sealion_dialogue_v1_data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open('/tmp/replycoach_sealion_dialogue_v1.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)))
        tar.add(path,arcname='recipe.json')
    print('Frozen',len(files),'files; public7994/400, project380/100, new development12.')

if __name__=='__main__':main()
