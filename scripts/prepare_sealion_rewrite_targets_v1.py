"""Matched old/revised target wording with fixed English and common casual replay."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from replycoach.generate.sealion_target_overlay_v1 import authored_pairs,schedule
OUT=ROOT/'artifacts/sealion_rewrite_targets_v1'
TEAM_SHA='03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7'


def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if p.exists() and p.read_text()!=text:raise FileExistsError(p)
    p.write_text(text);return p


def main():
    assert sha(ROOT/'data/dev/train.jsonl')==TEAM_SHA
    source=read('artifacts/sealion_style_v1/train.json');overlay=read('artifacts/sealion_rewrite_targets_v1/overlay.json')
    assert len(source)==745 and len(overlay)==120 and sum(r['changed'] for r in overlay)==100
    ids={r['item_id'] for r in overlay};by_id={r['item_id']:r for r in source}
    for row in overlay:
        assert row['item_id'].startswith('SYN-')
        for k in ['context','speakers','message','standard_anchor','response']:assert row[k]==by_id[row['item_id']][k]
    keep=lambda r:{k:r[k] for k in ['item_id','context','speakers','message','standard_anchor','response']}
    replay=[keep(r) for r in source if r['item_id'] not in ids]
    control=[keep(r) for r in overlay]
    revised=[dict(keep(r),response=r['new_response']) for r in overlay]
    assert len(replay)==625 and {r['item_id'] for r in replay}.isdisjoint(ids)
    assert {r['item_id'] for r in replay}|ids==set(by_id)
    authored=authored_pairs();legacy=read('artifacts/sealion_style_v1/validation.json')
    raw=read('artifacts/casual_sealion_mt5_v1/comparison.json');demo=[]
    for case in raw['comparison']:
        for model,out in case['outputs'].items():
            demo.append(dict(item_id='REWRITE-DEMO-'+case['item_id']+'-'+model,
                context=case['context'],speakers=case['speakers'],message=case['message'],standard_anchor=out['english'],
                scope='reused_casual_generated_anchor',anchor_provenance='Actual saved '+model+' English; may itself be contextually weak. No English repair.'))
    groups=dict(replay=replay,control_overlay=control,revised_overlay=revised,common=authored['common_train'],
        validation=authored['validation'],test=authored['test'],controls=authored['fidelity_controls'],legacy=legacy,demo=demo)
    key=lambda r:json.dumps([r['context'],r['speakers'],r['message'],r['standard_anchor']],sort_keys=True)
    train_keys={key(r) for r in replay+control+authored['common_train']}
    heldout=authored['validation']+authored['test']+authored['fidelity_controls']+legacy+demo
    assert not train_keys&{key(r) for r in heldout}
    assert len({key(r) for r in authored['validation']+authored['test']})==40
    paths={f'data/{name}.json':save('data/'+name+'.json',rows) for name,rows in groups.items()}
    paths['schedule.json']=save('schedule.json',schedule(625))
    guide=dict(style='Natural/chatty Singapore texting; moderate strength, varied phrasing.',
        preference_source='Default announced in optional style question; user can steer. No claim of independently native-rated gold.',
        preserve=['speaker/owner/recipient','people/places/numbers/timing','negation/completion/modality','decision and question/request function'],
        allow=['context-appropriate local word order','natural tense/aspect and subject omission','selective particles','local vocabulary where meaning fits','unchanged short replies when natural'],
        avoid=['mandatory lah/leh/Can','all targets using one syntactic template','word substitutions that change meaning','adding past events, commitments, places or ownership','copy penalties used as a naturalness score'])
    paths['style_guide.json']=save('style_guide.json',guide)
    audit=[dict(item_id=r['item_id'],english=r['standard_anchor'],old_target=r['response'],new_target=r['new_response'],
        changed=r['changed'],source_inputs_unchanged=True,review=r['review_criterion'],
        note=r['edit_note'],independent_human_review=False) for r in overlay]
    paths['target_audit.json']=save('target_audit.json',audit)
    # The unit tests verify the original artifacts as well as the runtime pools.
    paths['artifacts/sealion_style_v1/train.json']=ROOT/'artifacts/sealion_style_v1/train.json'
    paths['artifacts/sealion_rewrite_targets_v1/overlay.json']=OUT/'overlay.json'
    paths['protected_team.jsonl']=ROOT/'data/dev/train.jsonl'
    flat={
        'run_sealion_rewrite_targets_v1.py':'scripts/run_sealion_rewrite_targets_v1.py',
        'summarize_sealion_rewrite_targets_v1.py':'scripts/summarize_sealion_rewrite_targets_v1.py',
        'run_sealion_style_v1.py':'scripts/run_sealion_style_v1.py',
        'sealion_style_v1.py':'src/replycoach/generate/sealion_style_v1.py',
        'sealion_target_overlay_v1.py':'src/replycoach/generate/sealion_target_overlay_v1.py',
        'tests/test_sealion_target_overlay_v1.py':'tests/test_sealion_target_overlay_v1.py',
        'author_sealion_rewrite_targets_v1.py':'scripts/author_sealion_rewrite_targets_v1.py',
        'prepare_sealion_rewrite_targets_v1.py':'scripts/prepare_sealion_rewrite_targets_v1.py',
    }
    paths.update({name:ROOT/path for name,path in flat.items()})
    recipe=dict(experiment='sealion_rewrite_targets_v1',account='hm.teoh.2024',seed=27149,
        purpose='Isolate natural/chatty Singlish target wording with fixed English anchors. No English-generation update, copy penalty, lexicon reranker or rule repair.',
        source_bundle='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1',
        original_english_sha256='55a7f02626c08894bfdc4b44f68dacf86e8f4edafb7aa6bd78aead9162698913',
        initial_singlish_sha256='6ea7066deda4f4e9e5f4a9e070d46427536a0dac05d5a416864e97c2dc3f5db2',
        arms=['old_targets_control','revised_targets'],
        data=dict(original_eligible_pairs=745,unchanged_replay=625,overlay_contexts=120,revised_targets=100,unchanged_overlay_controls=20,
            shared_new_casual_pairs=20,new_validation=20,new_test=20,fidelity_controls=10,reused_old_validation=42,
            reused_generated_anchor_examples=12,reused_underlying_casual_conversations=6),
        matching='Same initial Singlish LoRA, contexts, English drafts, update count, per-step seed and example schedule. Both receive20shared new casual pairs. Only100overlay target strings differ. Target lengths/dropout positions and token/FLOP counts may differ. Overlay IDs excluded from replay so revised arm never also sees old target for those same inputs.',
        training=dict(updates=240,replay_per_step=5,common_per_step=1,overlay_per_step=2,
            replay_exposures=1200,common_exposures=240,overlay_exposures=480,each_overlay_visits=4,each_common_visits=12,
            learning_rate=1e-5,weight_decay=.01,clip_norm=1.,rank=8,alpha=16,dropout=.05,trainable_parameters=10485760,
            precision='BF16 model, FP32 reply CE',objective='Mean8example reply-only cross entropy. No auxiliary negative/copy/style-reward loss.'),
        snapshots=[0,60,120,240],
        selection='Minimum20newvalidation referenceCE including0, tieearlier. Old42CE separately reported, no old-loss retention guard. Generate20validation drafts at each snapshot for later side-by-side review. Test/control/demo outputs cannot select checkpoint. All snapshots and final240 retained; no promotion from loss alone.',
        inference=dict(num_beams=4,no_repeat_ngram_size=3,max_new_tokens=64,total_limit=384,
            anchors='Fixed supplied English drafts.12reused demonstrations use actual stored SEA/mT5English; other newcases use labelled assistant-authored English for isolated rewriting.'),
        evaluation='Selected/final rewrite outputs on20newtest+10fidelitycontrols+12reusedactualEnglishanchors+42oldvalidation=84; snapshotvalidation20each. Original also generated on same104total. Naturalness, semantic fidelity and tone judged separately; unchanged text and counts of particles/copies are diagnostics, not quality scores.',
        limitations='Assistant-authored style targets and audit, not independently human-reviewed gold. Newcases are illustrative, oldvalidation/demos reused. Common20newpairs may improve both arms; improvements over original cannot all be attributed to100targetrevisions. One seed and learning rate; lowerCE need not mean more natural Singlish.',
        protected_team_sha256=TEAM_SHA,original_dataset_files_unchanged=True,default_changed=False,locked_test_read=False,
        payload_manifest={name:sha(path) for name,path in paths.items()})
    save('recipe.json',recipe);(ROOT/'experiments/sealion_rewrite_targets_v1_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open(OUT/'payload.tgz','w:gz') as tar:
        for name,p in paths.items():tar.add(p,arcname=name,recursive=False)
        tar.add(OUT/'recipe.json',arcname='recipe.json',recursive=False)
    print(json.dumps(dict(data=recipe['data'],payload_sha256=sha(OUT/'payload.tgz'))))


if __name__=='__main__':main()
