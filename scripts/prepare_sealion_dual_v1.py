"""Freeze original dual-style supervision and new development evaluation."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/evaluation/sealion_dual_v1'
EXAMPLES=[
 (['My shift ends at 6:35 and I cannot leave early.','We could have dinner afterwards.'],['user','peer'],'Can we meet for dinner at 5:55?','Respect earliest departure; a refusal, later alternative or clarification is valid.'),
 (['My lunch budget today is eleven dollars.','We can find something nearby.'],['user','peer'],'Want the twenty-seven-dollar lunch set?','Do not accept an option incompatible with stated budget without resolving that constraint.'),
 (['I am free on Friday, but Saturday is unavailable.','I will check the schedule.'],['user','peer'],'Can you attend the rehearsal on Saturday?','Saturday explicitly unavailable; alternatives/refusals allowed.'),
 (['You still have my USB cable.','Yes, it is in my bag.'],['user','peer'],'Can I return your cable on Monday?','PEER returns USER cable; avoid ownership/returner reversal.'),
 (['I still have your hard-copy revision notes.','I need them for class.'],['user','peer'],'Could you return my notes before nine?','USER returns PEER notes, or gives a coherent refusal/alternative.'),
 (['I saved the practice clip on my laptop.','I missed that practice.'],['peer','user'],'Would you like me to share the clip with you?','PEER offers clip to USER; respond without reversing sender.'),
 (['I have the interview notes on my tablet.','I missed the interview.'],['user','peer'],'Could you send me those notes tonight?','USER has notes and is asked to send to PEER; no role reversal.'),
 (['I already booked our tutorial room this morning.','Thanks for arranging that.'],['user','peer'],'Have you booked the room yet?','Booking completed, not first future booking.'),
 (['I have not started proofreading the report.','We still have until Friday.'],['user','peer'],'Is the proofread report ready?','Not started; do not claim completion.'),
 (['I am waiting outside seminar room 4A.','I am coming upstairs.'],['user','peer'],'Which room are you outside?','Answer known room 4A or a coherent clarification, not an invented different room.'),
 (['I left my purple water bottle in the studio.','There are a few bottles here.'],['user','peer'],'What colour is your bottle?','Purple is established; no colour change.'),
 (['I have not checked the cafeteria weekend hours.','Holiday hours might be different.'],['user','peer'],'Does it close at seven?','Hours unknown; uncertainty/checking/deferral allowed.'),
 (['I am still in the tutorial.','I am at the coffee stall.'],['user','peer'],'What drink should I get you?','PEER buys for USER; any plausible preference/refusal/clarification valid.'),
 (['I am free after class.','There is a theatre show tonight.'],['user','peer'],'Would you like to join us?','Any context-compatible acceptance, refusal or alternative; do not demand unique intent.'),
 (['I have been doing most of the group project alone.','Have the others been helping?'],['peer','user'],'Not really, and I am feeling overwhelmed.','Respond supportively to PEER overwhelm; speaker roles consistent.'),
 (['My interview results came out today.','How did it go?'],['peer','user'],'I did not get the placement.','Acknowledge disappointment, no congratulations for success.'),
 (['My wrist is injured, so I cannot carry heavy things.','Hope it gets better.'],['user','peer'],'Can you help carry these heavy crates?','Respect explicit physical limitation.'),
 (['I can meet after my lab.','We could work on the poster.'],['user','peer'],'Meet me there when you finish.','Location unspecified; clarify/propose rather than invent a supposedly agreed place.'),
 (['I am studying in the library but it is noisy here.','We could find somewhere quieter.'],['user','peer'],'Would you prefer to move somewhere else?','Any coherent preference/alternative, no exact reference wording required.'),
 (['I submitted the poster file yesterday.','I spotted the wrong date in it.'],['user','peer'],'I corrected it and sent you the updated file.','Acknowledge completed PEER correction; review/thanks are valid.'),
]


def signature(row):return json.dumps([row['context'],row['speakers'],row['message']],separators=(',',':')).lower()


def main():
    FOLDER.mkdir(exist_ok=True,parents=True)
    inputs=[];guidance=[]
    for i,(context,speakers,message,note) in enumerate(EXAMPLES,1):
        key=f'SEADUAL-{i:03d}';inputs.append(dict(item_id=key,context=context,speakers=speakers,message=message))
        guidance.append(dict(item_id=key,expectation=note,label_source='assistant development guidance, not independent human gold'))
    known=set()
    for split in ['train','validation']:
        for line in (ROOT/f'data/dev/{split}.jsonl').read_text().splitlines():
            r=json.loads(line);known.add(signature(dict(context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'])))
    for folder in ['mt5_grounding_audit_v1','mt5_target_repair_v1','component_baseline_v1']:
        known.update(signature(r) for r in json.loads((ROOT/f'data/evaluation/{folder}/inputs.json').read_text()))
    assert len({signature(r) for r in inputs})==20 and not {signature(r) for r in inputs}&known
    (FOLDER/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (FOLDER/'review_guidance.json').write_text(json.dumps(guidance,indent=2)+'\n')
    paired_notes=dict(policy='Both original targets copied exactly, no synthetic additions, no source-role corrections, no style rewrites. This preserves the requested190 pairs but does not certify their semantic quality.',
        examples=[dict(item_id='ITEM-C-047',observation='Original STANDARD says meet after class; original LOCAL mentions taking MRT together. Added detail means targets are not strict paraphrases.'),
                  dict(item_id='ITEM-B-017',observation='Incoming/history have inconsistent sending directions; original targets retained and source ambiguity must not be treated as clean factual gold.')],
        prior_audit='experiments/mt5_training_lineage_audit_v1.json; original versus revised target differences are known. Revisions are deliberately not silently mixed into this run.')
    (FOLDER/'target_notes.json').write_text(json.dumps(paired_notes,indent=2)+'\n')
    vendor=ROOT/'src/replycoach/vendor/sealion_v1';provenance=json.loads((vendor/'provenance.json').read_text())
    files=[*sorted(ROOT.glob('src/**/*.py')),ROOT/'src/replycoach/__init__.py',ROOT/'src/replycoach/schema.py',ROOT/'src/replycoach/tokenize.py',ROOT/'src/replycoach/generate/__init__.py',ROOT/'src/replycoach/vendor/__init__.py',
        *[ROOT/'src/replycoach/generate'/n for n in ['sealion_v1.py','sealion_data.py','sealion_dual_v1.py']],
        *[p for p in sorted(vendor.iterdir()) if p.is_file()],
        *[ROOT/'scripts'/n for n in ['prepare_sealion_dual_v1.py','run_sealion_dual_v1.py','v5_sealion_dual_v1.sbatch']],
        ROOT/'tests/test_sealion_dual_v1.py',ROOT/'tests/test_sealion_data.py',ROOT/'tests/test_sealion_v1_compat.py',
        ROOT/'data/dev/train.jsonl',ROOT/'data/dev/validation.jsonl',*sorted(FOLDER.glob('*.json'))]
    files=list(dict.fromkeys(files))
    recipe=dict(experiment='sealion_dual_v1',repo_id=provenance['repo_id'],revision=provenance['revision'],weights_sha256=provenance['weights']['lfs']['sha256'],
        train_conversations=190,train_targets=380,validation_conversations=50,validation_targets=100,styles=['STANDARD','LOCAL'],
        seed=425,lora=dict(r=8,alpha=16,dropout=.05,target_modules=['Wqkv','out_proj','up_proj','down_proj']),
        precision='BF16 base/frozen; PEFT LoRA adapters; FP32 shifted reply-only CE',learning_rate=1e-4,weight_decay=.01,microbatch=1,gradient_accumulation=8,max_epochs=12,patience=2,sequence_limit=384,
        selection='Minimum equal-style-average validation token CE among trained epochs1+. Epoch0 untrained base measured separately; no untrained prompted model qualifies as final candidate.',
        decoding='Deterministic greedy, legacy tuple KV cache tested against full-context logits, EOS or64 new tokens or next serialized speaker turn. Keep raw text and termination reason; capped/empty outputs stay visible.',
        evaluations=['step0_base diagnostic on20 new development conversations, both styles','selected trained adapter on same20 and50 reused validation conversations, both styles'],
        evaluation_policy='Automatic contextual usefulness first, then STANDARD/LOCAL meaning consistency and naturalness; no intent-label/exact-reference requirement.20 authored cases are structurally related development material, not human gold. Never select checkpoints on these outputs.',
        data_policy=paired_notes,comparison_scope='Train SEA-LION first and preserve paired outputs; mT5 comparison can use matching development inputs later. Different pretraining histories/architectures, not an isolated parameter-size experiment.',
        gpu='RTX3090',default_changed=False,files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    path=ROOT/'experiments/sealion_dual_v1_recipe.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open('/tmp/replycoach_sealion_dual_v1.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)))
        tar.add(path,arcname='recipe.json')
    print('Frozen380 training targets,100 validation targets and20 fresh development conversations.')

if __name__=='__main__':main()
