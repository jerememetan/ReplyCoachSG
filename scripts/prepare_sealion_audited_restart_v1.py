"""Freeze the user-calibrated restart without modifying source training/validation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/sealion_audited_restart_v1'
REMOTE='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_audited_restart_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def case(ident,context,speakers,message,standard,local):
    return dict(item_id=ident,context=context,speakers=speakers,message=message,standard=standard,local=local,reference_provenance='Assistant-authored following five user style preferences, not individually human-approved.')

def main():
    assert not OUT.exists();OUT.mkdir()
    audit=ROOT/'artifacts/target_style_audit_v1';summary=read(audit/'summary.json')
    for p,v in summary['source_preservation'].items():assert sha(ROOT/p)==v['expected'],p
    spec=importlib.util.spec_from_file_location('helper',ROOT/'src/replycoach/generate/sealion_audited_restart_v1.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    records=[json.loads(l) for l in (audit/'combined_train_PROPOSED.jsonl').read_text().splitlines()]
    flags=read(audit/'flagged_rows.json');excluded={r['item_id'] for r in flags};assert len(excluded)==22
    train=h.filter_training(records,excluded);assert len(train)==768
    old=[h.from_record(json.loads(l)) for l in (ROOT/'data/dev/validation.jsonl').read_text().splitlines()]
    # Context-supported ellipsis is eligible. Two remaining pairs add a completed reservation beyond finding seats.
    extra={'ITEM-C-010':'LOCAL says a seat has been reserved, whereas English only says two seats were found.',
           'ITEM-E-029':'LOCAL says a seat was reserved, whereas English only says one was found.'}
    rewrite=[r for r in train if r['item_id'] not in extra]
    old_audit=read(ROOT/'data/training_expansion/sealion_rewrite_v1/pair_audit.json')
    old_ok={r['id'] for r in old_audit['items'] if r['split']=='validation' and r['eligible']}
    old_rewrite=[r for r in old if r['item_id'] in old_ok];assert len(old_rewrite)==42
    vals=[
    (['The picnic photos are in my album.','Send me the link?'],['peer','user'],'Sent it on Signal already.','Thanks, I will check Signal.','thanks, i check'),
    (['I can lend you my calculator for the lab tomorrow.'],['user'],'rmb bring it tmr hor','Okay, I will bring it tomorrow.','okok, will do'),
    (['Want to come for the board games tonight?'],['user'],'sorry, too tired today, next time?','No worries, we can play another time.','aw, next time bah'),
    (['Meet at the climbing gym at 6:40?','Okay.'],['peer','user'],'6:40 still on right?','Yes, six forty is still on.','yep'),
    (['I am using your charging cable.'],['user'],'can return after you charge finish?','Yes, I will return it after charging.','can, charge finish then return'),
    (['I am waiting beside the medical centre.'],['user'],'where you waiting ah?','I am beside the medical centre.','beside medical centre'),
    (['Getting drinks, what you want?'],['peer'],'lime juice or barley?','Lime juice, please.','lime juice pls'),
    (['I uploaded the sketch earlier.'],['user'],'sketch uploaded already?','Yes, I uploaded it earlier.','uploaded already'),
    (['I have not started editing the audio yet.'],['user'],'audio edit finish anot?','Not yet, I have not started editing.','haven’t started editing yet'),
    (['I have my umbrella with me.'],['user'],'can lend me for the walk home?','Sure, you can use my umbrella.','can, use mine first'),
    (['I forgot my umbrella.','I have a spare one.'],['user','peer'],'you want borrow mine?','Yes please, could I borrow yours?','yes pls, lend me?'),
    (['I cannot leave work until 7pm.'],['user'],'meet at 6:20 can?','I cannot meet at six twenty. Could we meet after seven?','6:20 cannot, after 7 can?'),
    (['I can spend six dollars.','The meal is twelve and the snack is three.'],['user','peer'],'which one you want?','The meal is over my budget. I will take the snack instead.','meal too ex, snack instead'),
    (['Hi prof, when will the feedback be ready?'],['user'],'I can send it by Friday, is that alright?','That is fine, thank you professor.','okies thanks prof!'),
    (['Do you know the office opening hours?'],['peer'],'open on Sunday anot?','I am not sure, I will check the directory.','not sure, i check directory'),
    (['We are using room R09.','Actually it has moved to R12.'],['peer','peer'],'see you at R12 then','Okay, I will go to R12.','okok, i go R12'),
    (['I did not get the scholarship.'],['peer'],'quite upset tbh','I am sorry to hear that. Would you like to talk about it?','aiyo, wanna talk about it?'),
    (['I sent a few messages this morning.'],['peer'],'did you see that msg?','Which message do you mean?','which msg ah?'),
    (['I can reserve only two seats.'],['user'],'can reserve three for us?','I can only reserve two. What should we do about the third?','two only, third one how?'),
    (['My coat is black, not navy.'],['user'],'which coat should i bring you?','Please bring my black coat.','black coat pls'),
    ]
    val=[case(f'AUDIT-VAL-{i:02d}',*r) for i,r in enumerate(vals,1)]
    tests=[
    (['I am sneezing badly, staying home today.','aiyo'],['peer','user'],'you guys go arcade first, send me clips later','Sure, I will send you clips later.','okok, will do'),
    (['I left the snacks at your door.'],['peer'],'saw them already?','Yes, I saw them. Thank you!','saw already, thanks!'),
    (['Want to go for a walk by the reservoir?','sounds good'],['peer','user'],'eh tmr 5:15 still can right?','Yes, five fifteen still works.','yep'),
    (['I finished sorting the playlist for our drive.'],['user'],'settled the playlist?','Yes, it is ready.','settled liao'),
    (['I can bring my speakers for the picnic.'],['user'],'ok rmb charge them first ya','Okay, I will charge them first.','okok, will do'),
    (['I cannot make it for supper, need sleep early.'],['peer'],'ps ah next time jio again','No worries, I will invite you next time.','no worries, next time jio'),
    (['The old power socket is spoilt.','I found a working one near the window.'],['peer','user'],'can share that socket with me?','Sure, you can share it with me.','can ah'),
    (['I left my headphones at home.','I have a spare pair.'],['user','peer'],'want use mine for the recording?','Yes please, could I use your spare pair?','yes pls, thanks'),
    (['The backpack you lent me is in my room.'],['user'],'rmb pass me my bag after dinner','Okay, I will return your bag after dinner.','okok, dinner finish pass you'),
    (['I am outside the florist.','I am at the taxi stand.'],['user','peer'],'where you at now?','I am outside the florist.','outside florist'),
    (['I have to leave by 8:10pm.'],['user'],'movie starts 8, two hours, steady?','I cannot stay for that movie. I need to leave by eight ten.','cannot eh, 8:10 need leave'),
    (['I only have nine dollars for food.','The noodles are fourteen dollars.'],['user','peer'],'get those noodles tgt?','That is over my budget. Could we get something cheaper?','over budget, cheaper one can?'),
    (['I have finished the diagram but not the explanation.'],['user'],'whole thing done?','Not yet, I still need to write the explanation.','not yet, explanation haven’t done'),
    (['The coffee shop closes at nine.','I can only arrive at nine fifteen.'],['peer','user'],'meet there when you reach?','It will be closed when I arrive. Could we meet somewhere else?','by then close already, somewhere else can?'),
    (['Hi prof, could I check when consultations start?'],['user'],'I will put the booking slots up on Wednesday.','Thank you professor, I will check on Wednesday.','thanks prof, Wed i check'),
    (['I saved your scarf, it is on my chair.'],['user'],'is my scarf still with you?','Yes, it is on my chair.','ya, on my chair'),
    (['Can you help me get a drink?'],['peer'],'either lemon tea or plain water','Which one would you prefer?','which one you want?'),
    (['I know the charger you left is yours.'],['user'],'send me a photo to confirm?','Sure, I will send you a photo.','okok, i send'),
    (['The tutorial got shifted to lab L8.'],['peer'],'still going to L3?','No, I will go to L8 instead.','no, L8 now'),
    (['I may have left my transit card at the bakery.'],['user'],'found it already?','Not yet, I will check the bakery.','not yet, i check bakery'),
    (['That practice presentation went so badly.'],['peer'],'damn demoralised now','Do you want to take a break and go through it together later?','take a break first? later go through tgt'),
    (['Are you free to help me move the chairs?'],['peer'],'only need help for 10 min','Sure, when do you need me?','can, when ah?'),
    (['The parcel in the blue box belongs to you.'],['peer'],'blue or brown box should i bring?','Please bring the blue box.','blue one pls'),
    (['I have not checked whether the studio is open.'],['user'],'open on public holiday?','I am not sure. I will check their announcement.','dunno, i check announcement'),
    ]
    test=[case(f'AUDIT-EVAL-{i:02d}',*r) for i,r in enumerate(tests,1)]
    h.check_splits({'train':train,'old_validation':old,'style_validation':val,'fresh_evaluation':test})
    reused=read(ROOT/'artifacts/user_flu_sealion_mt5_v1/inputs.json')[0];reused=dict(reused,scope='reused_user_diagnostic')
    evaluation=[dict(r,scope='fresh_assistant_evaluation') for r in test]+[reused]+[dict(r,scope='reused_old_validation') for r in old]
    save(OUT/'data/task_train.json',train);save(OUT/'data/rewrite_train.json',rewrite)
    save(OUT/'data/old_validation.json',old);save(OUT/'data/old_rewrite_validation.json',old_rewrite)
    save(OUT/'data/style_validation.json',val);save(OUT/'data/evaluation.json',evaluation)
    save(OUT/'data/exclusions.json',dict(source_flags=flags,excluded_from_both=sorted(excluded),additional_rewrite_only=extra,
         policy='Exclude unresolved source concerns from training copies, not source files. New audit accepts context-dependent acknowledgments; do not carry forward older literal-omission exclusions blindly.'))
    shutil.copy2(ROOT/'data/dev/train.jsonl',OUT/'protected_train.jsonl');shutil.copy2(ROOT/'data/dev/validation.jsonl',OUT/'protected_validation.jsonl')
    stage=OUT/'stage';stage.mkdir();shutil.copytree(ROOT/'src',stage/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (stage/'tests').mkdir();shutil.copy2(ROOT/'tests/test_sealion_audited_restart_v1.py',stage/'tests')
    (stage/'scripts').mkdir()
    for name in ['run_sealion_audited_restart_v1.py','summarize_sealion_audited_restart_v1.py']:
        shutil.copy2(ROOT/'scripts'/name,stage/'scripts'/name)
    shutil.copytree(OUT/'data',stage/'data')
    for name in ['protected_train.jsonl','protected_validation.jsonl']:shutil.copy2(OUT/name,stage/name)
    source_files={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
    recipe=dict(experiment='sealion_audited_restart_v1',account='hm.teoh.2024',remote_root=REMOTE,seed=425,
      base_path='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion3b_probe_v1/model',base_weight_sha256='acba6d6560f61ac72cc95feca5c89f6aabbbf1e9e89c18440237331fcb5ae107',
      public_adapter='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_full_v1/artifacts/full_public_adapter',
      public_hashes={'adapter_model.safetensors':'b0dfd6b9e70ba93b3dcbd9a0b76fd35800848fee1508f5c4e85937892acfd36d','adapter_config.json':'f5bb26bc63c376e64ed060f54cc87a44ece14a7328f064d02abc487aa75578bc'},
      original_bundle='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1',original_manifest_sha256='1cb24b55984b73c20166331d4d872895d00d9d4854b2a2b7c8f5245453db7f1a',
      task=dict(conversations=len(train),targets=2*len(train),original_conversations=sum(not r['item_id'].startswith('SYN-') for r in train),synthetic_conversations=sum(r['item_id'].startswith('SYN-') for r in train),learning_rate=1e-4,max_epochs=8,patience=2,selection='0.5 old50 equal-style CE +0.5 new20 equal-style CE, including epoch0; reference likelihood is not quality.'),
      rewrite=dict(pairs=len(rewrite),learning_rate=3e-5,max_epochs=8,patience=2,selection='0.5 old42 rewrite tokenCE +0.5 new20 rewrite tokenCE, including epoch0; no output-driven promotion.'),
      common=dict(precision='BF16 frozen base and FP32 trainable LoRA',effective_batch=8,microbatch=1,rank=8,alpha=16,dropout=.05,trainable_parameters=10485760,weight_decay=.01),
      inference=dict(num_beams=4,no_repeat_ngram_size=3,max_new_tokens=64,length_penalty=1.,total_budget=384),
      validation=dict(original_conversations=50,original_targets=100,rewrite_eligible_original=42,new_style_conversations=20,original_file_unchanged=True),
      evaluation=dict(new_assistant_conversations=24,reused_user=1,reused_original_validation=50,total=75,fixed_reference_rewrite_diagnostics=24,freeze_before_training=True),
      calibration='User selected A for all5. Remaining audit targets and new validation/evaluation references are assistant authored, not human gold.',
      exclusions='22 source concerns excluded from both stages;2 extra original seat-reservation pairs excluded from rewrite only. Sources unchanged.',
      source_protected_hashes={'protected_train.jsonl':sha(OUT/'protected_train.jsonl'),'protected_validation.jsonl':sha(OUT/'protected_validation.jsonl')},
      source_audit_summary_sha256=sha(audit/'summary.json'),payload_manifest=source_files,
      comparison='One new pipeline against existing original repair25. Different training histories prevent attributing changes solely to target wording. No duplicate old-target control, public retraining, classifier, planner, reranker, copy penalty or posthoc wording replacement.',
      limitations='One seed, assistant-authored/edited data and related everyday scenarios; original validation heavily reused. Lower CE, fewer words, particles or copies are not quality measures.',default_changed=False)
    save(OUT/'recipe.json',recipe);save(stage/'recipe.json',recipe);save(ROOT/'experiments/sealion_audited_restart_v1_recipe.json',recipe)
    for name,argument,gpu,time in [('preflight','preflight',True,'02:00:00'),('task','task',True,'04:00:00'),('rewrite','rewrite',True,'04:00:00'),('summary','',False,'00:15:00')]:
        command='scripts/run_sealion_audited_restart_v1.py --stage '+argument if name!='summary' else 'scripts/summarize_sealion_audited_restart_v1.py'
        batch='\n'.join(['#!/bin/bash',f'#SBATCH --job-name=sea-audit-{name}','#SBATCH --partition=student','#SBATCH --account=cs701','#SBATCH --qos=studentqos','#SBATCH --cpus-per-task=4','#SBATCH --mem=32G',*(['#SBATCH --gres=gpu:l40s:1','#SBATCH --constraint=l40s'] if gpu else []),f'#SBATCH --time={time}','#SBATCH --output=slurm-%x-%j.out','set -euo pipefail','module purge','module load Python/3.13.1-GCCcore-13.3.0',f'cd {REMOTE}',f'export PYTHONPATH="$PWD/src:{recipe["original_bundle"]}/deps" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false',f'/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python -u {command}',''])
        (OUT/(name+'.sbatch')).write_text(batch);(stage/(name+'.sbatch')).write_text(batch)
    with tarfile.open(OUT/'payload.tgz','w:gz') as tar:
        for p in sorted(stage.rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(stage)),recursive=False)
    save(OUT/'package.json',dict(payload_sha256=sha(OUT/'payload.tgz'),bytes=(OUT/'payload.tgz').stat().st_size,files=len([p for p in stage.rglob('*') if p.is_file()])))
    print(json.dumps(dict(task=len(train),rewrite=len(rewrite),style_validation=len(val),evaluation=len(evaluation),package=read(OUT/'package.json')),indent=2))

if __name__=='__main__':main()
