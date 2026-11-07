"""Freeze matched target-repair training and a small fresh usefulness evaluation."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/evaluation/mt5_target_repair_v1'
EXAMPLES=[
 ('I am going to the minimart after class.','I am still in a meeting.','Could you pick up a tea for me?','USER buys for PEER; do not reverse the buyer. Accept, refuse, or clarify sensibly.'),
 ('I am still in the seminar.','I am at the juice stall now.','What should I get you?','PEER offers to buy for USER. Any plausible choice/refusal/clarification is valid.'),
 ('I have the field-trip photos on my phone.','I did not take any photos.','Could you send me those photos?','USER has photos; avoid reversing sender/recipient.'),
 ('I could not download the registration pack.','I have a copy in my email.','Would you like me to forward it to you?','PEER offers the pack to USER; do not claim USER is sending it instead.'),
 ('You still have my USB drive.','Yes, it is in my laptop bag.','Can I return your USB drive tomorrow?','PEER holds USER drive; grant/refuse/clarify return permission without reversing ownership.'),
 ('I still have your whiteboard marker.','I need it for the lesson.','Could you return my marker before the lesson?','USER holds PEER marker; respond to return request with correct actors.'),
 ('You said you would bring your HDMI cable.','Yes, I will pack it tonight.','You are bringing the HDMI cable, right?','Earlier PEER promised their cable, now asks USER. Clarification or a coherent new offer is valid; do not require a unique hidden intent.'),
 ('I have not begun writing my lab report.','The submission is on Friday.','Is your lab report finished?','Not begun, so do not claim completed.'),
 ('I already uploaded the practice video.','I will look for the link.','Do you still need to upload the video?','Already uploaded; respect completed state.'),
 ('I only have nine dollars for dinner.','There are a few places nearby.','Want to get the twenty-two-dollar meal?','Do not accept the meal as if compatible with nine-dollar budget; refusals/alternatives/clarification allowed.'),
 ('I cannot leave the office until 4:40.','We could get coffee afterwards.','Shall we meet for coffee at 4:15?','4:15 is before earliest departure; do not agree without resolving conflict.'),
 ('I have not checked when the robotics room opens.','Holiday hours may be different.','Does it open at eight?','Opening time unknown; no unsupported certainty. Checking/deferral/clarification valid.'),
]


def signature(r):
    return json.dumps([r['context'],r['speakers'],r['message']],separators=(',',':')).lower()


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    rows=[];notes=[]
    for i,(user,peer,message,note) in enumerate(EXAMPLES,1):
        key=f'TARGETREPAIR-{i:03d}'
        rows.append(dict(item_id=key,context=[user,peer],speakers=['user','peer'],message=message))
        notes.append(dict(item_id=key,expectation=note,label_source='assistant-authored factual guidance, not independently human reviewed'))
    seen=set()
    for split in ['train','validation']:
        for line in (ROOT/f'data/dev/{split}.jsonl').read_text().splitlines():
            r=json.loads(line);seen.add(signature(dict(context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'])))
    for path in ['data/evaluation/mt5_grounding_audit_v1/inputs.json','data/training_expansion/mt5_act_follow_v1/train.json','data/training_expansion/mt5_act_follow_v1/validation.json']:
        seen.update(signature(r) for r in json.loads((ROOT/path).read_text()))
    assert len({signature(r) for r in rows})==12 and not {signature(r) for r in rows}&seen
    (FOLDER/'inputs.json').write_text(json.dumps(rows,indent=2)+'\n')
    (FOLDER/'review_guidance.json').write_text(json.dumps(notes,indent=2)+'\n')
    protocol=dict(primary='Automatic English reply usefulness: factual consistency, response to latest message, coherent roles and practical sendability. Any valid preference/refusal/alternative is allowed, independent of intent labels or reference wording.',
        secondary='LOCAL meaning preservation and naturalness scored separately; English correctness does not imply LOCAL correctness.',
        scope='12 newly frozen assistant-authored development conversations. Similar semantics exist in training. Not independent human gold or original locked product test. Prior28 audit cases are reused diagnostics, not fresh tests.',
        selection='Epoch selection only on fixed50 original and16 paired validation targets, equally weighted CE. No selection on these new12 outputs or reused28 diagnostics.',
        provenance='Five repaired targets are assistant corrections/clarifications, not independently human reviewed. Both arms otherwise identical238-row data.',
        seeds=[425,426],stop_rule='No automatic further training/default replacement after results. Report mixed effects, uncertainty and failures, not merely reference loss or seed counts as independent conversations.')
    (FOLDER/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    files=[*sorted(ROOT.glob('src/**/*.py')),
        *[ROOT/'scripts'/n for n in ['prepare_mt5_target_repair_v1.py','run_mt5_target_repair_v1.py','v5_mt5_target_repair_v1.sbatch']],
        ROOT/'tests/test_mt5_target_repair_v1.py',ROOT/'data/dev/validation.jsonl',
        ROOT/'data/training_expansion/mt5_act_follow_v1/validation.json',
        ROOT/'data/evaluation/mt5_grounding_audit_v1/inputs.json',
        *sorted(ROOT.glob('data/training_expansion/mt5_grounding_repair_v1/*')),
        *sorted(FOLDER.glob('*.json'))]
    recipe=dict(experiment='mt5_target_repair_v1',training=True,default_changed=False,arms=['control','repair'],seeds=[425,426],rows_per_arm=238,
        changed_target_ids=json.loads((ROOT/'data/training_expansion/mt5_grounding_repair_v1/manifest.json').read_text())['changed_target_ids'],
        architecture='Unchanged full mT5 STANDARD encoder-decoder in both arms. No reply-act head, conditioning or classification/ranking loss in either arm. Old auxiliary-loss runs are historical references only.',
        base_checkpoint='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_standard_val_v1/artifacts/full_lr3e4/best',
        rewrite_checkpoint='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_hybrid_inference_v1/checkpoints/local_rewrite',
        learning_rate=3e-5,weight_decay=.01,batch_size=2,gradient_accumulation=4,max_epochs=6,patience=2,max_source=192,max_target=48,
        selection='.5 original50 validation token CE + .5 paired16 validation token CE; epoch0 eligible; strict improvement, patience2',
        evaluation=protocol,files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    p=ROOT/'experiments/mt5_target_repair_v1_recipe.json';p.write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open('/tmp/replycoach_mt5_target_repair_v1.tar.gz','w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)))
        tar.add(p,arcname='recipe.json')
    print('Frozen12 fresh development conversations; matched238-row arms,2 paired seeds; snapshot ready.')

if __name__=='__main__':main()
