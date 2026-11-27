"""Freeze original-preserving, multi-reference conversational supervision."""
import hashlib
import json
from pathlib import Path
import random
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from replycoach.generate.sealion_creative_v1 import dataset,schedule
OUT=ROOT/'artifacts/sealion_creative_v1'
TEAM_SHA='03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if p.exists() and p.read_text()!=text:raise FileExistsError(p)
    p.write_text(text);return p


def main():
    assert sha(ROOT/'data/dev/train.jsonl')==TEAM_SHA
    expanded=ROOT/'data/training_expansion/generated600_repair_v1/combined_train.jsonl'
    assert expanded.read_bytes().startswith((ROOT/'data/dev/train.jsonl').read_bytes())
    original=[json.loads(x) for x in expanded.read_text().splitlines() if x.strip()]
    ordinary=[dict(item_id=r['id'],context=[t['text'] for t in r['session_memory']],speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],positive=r['replies']['standard']) for r in original]
    assert len(ordinary)==790
    groups=dataset()
    legacy=read('artifacts/sealion_turn_repair_v1/data/legacy.json')
    natural=read('artifacts/mt5_cocktail_v1/data/natural_challenge.json')
    # These already inspected mT5 inputs are development transfer checks, not fresh tests.
    natural=[dict(r,scope='reused_mt5_natural_diagnostic',criterion=r['review_constraints']) for r in natural]
    for row in natural:
        if row['item_id']=='COCKTAIL-NATURAL-10':
            row['criterion']='User initiated the call. A newly proposed purpose or text-first action is allowed. Preserve speaker direction and do not invent a prior shared agreement.'
    diagnostics=read('artifacts/mt5_cocktail_v1/data/diagnostics.json')
    diagnostics=[dict(r,criterion='Propose a useful next reply; preserve stated purpose if present, caller direction and Friend unavailability.') for r in diagnostics]
    key=lambda r:json.dumps([r['context'],r['speakers'],r['message']],sort_keys=True)
    assert not {key(r) for r in groups['train']} & {key(r) for r in ordinary+legacy+natural+diagnostics+groups['validation']+groups['challenge']}
    groups.update(ordinary=ordinary,legacy=legacy,natural=natural,diagnostics=diagnostics)
    paths={f'data/{name}.json':save(f'data/{name}.json',rows) for name,rows in groups.items()}
    paths['protected_team.jsonl']=ROOT/'data/dev/train.jsonl'
    paths['source_records.jsonl']=expanded
    paths['schedule.json']=save('schedule.json',schedule(790))
    review=[dict(item_id=r['item_id'],family=r['family'],criterion=r['criterion'],alternatives_reviewed=3,
        negative_reviewed=r['negative_reason'],status='assistant_authored_and_assistant_reviewed_not_human_gold') for r in groups['train']+groups['validation']+groups['challenge']]
    paths['target_audit.json']=save('target_audit.json',review)
    flat={
        'run_sealion_creative_v1.py':'scripts/run_sealion_creative_v1.py',
        'summarize_sealion_creative_v1.py':'scripts/summarize_sealion_creative_v1.py',
        'prepare_sealion_creative_v1.py':'scripts/prepare_sealion_creative_v1.py',
        'run_sealion_turn_repair_v1.py':'scripts/run_sealion_turn_repair_v1.py',
        'sealion_style_v1.py':'src/replycoach/generate/sealion_style_v1.py',
        'sealion_creative_v1.py':'src/replycoach/generate/sealion_creative_v1.py',
        'sealion_creative_sampling_v1.py':'src/replycoach/generate/sealion_creative_sampling_v1.py',
        'tests/test_sealion_creative_v1.py':'tests/test_sealion_creative_v1.py',
    }
    paths.update({name:ROOT/path for name,path in flat.items()})
    recipe=dict(experiment='sealion_creative_v1',account='hm.teoh.2024',seed=12081,
        goal='Useful creative conversational drafts with optional new intentions/actions, preserving known facts and roles. No abstention objective or refusal gate.',
        source_bundle='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1',
        source_english_sha256='55a7f02626c08894bfdc4b44f68dacf86e8f4edafb7aa6bd78aead9162698913',
        frozen_singlish_sha256='6ea7066deda4f4e9e5f4a9e070d46427536a0dac05d5a416864e97c2dc3f5db2',
        initial_weights='Reset original repair25 English LoRA for each arm; no abstention checkpoint used.',
        arms=['single_reference','multiple_references','multiple_plus_contrast'],
        data_counts={k:len(v) for k,v in groups.items()},
        new_references=dict(train_english=300,train_singlish=300,validation_english=60,challenge_english=60),
        training=dict(updates=300,ordinary_per_step=4,new_contexts_per_step=4,new_context_visits=12,
            each_alternative_visits_multi=4,ordinary_exposures=1200,new_context_exposures=1200,
            learning_rate=1e-5,weight_decay=.01,clip_norm=1.,trainable_parameters=10485760,
            base_parameters=3178255360,rank=8,alpha=16,dropout=.05,precision='BF16 model, FP32 CE'),
        objective='0.5*mean4ordinaryCE +0.5*mean4newPositiveCE +contrast_weight*mean4relu(0.2+positiveCE-negativeCE). contrast_weight0/0/0.1. Single-reference always alternative0; multi arms rotate alternatives equally. All arms forward/backpropagate negative graphs; inactive contrast has zero coefficient.',
        matching='Same original adapter, contexts, ordinary schedule, updates, LR and per-step dropout seed. Single/multi target lengths differ. Multi and multi+contrast have identical selected positives/negatives and shapes; only contrast coefficient differs. Primary single-vs-multi compares reference diversity at equal context exposures.',
        snapshots=[0,75,150,300],
        selection='Minimum0.5old50tokenCE+0.5mean20new-context mean3referenceCE, including step0; ties earlier. No negative ranking, sampling diversity, challenge or user outputs used for selection. Final300 also evaluated.',
        singlish_policy='Original repair25 Singlish adapter frozen for this English-continuation pilot. New paired Singlish references retained for review and future adaptation, not trained in this stage.',
        inference=dict(beam=4,no_repeat_ngram_size=3,max_new_tokens=64,total_limit=384,
            sampling=dict(candidates=3,temperature=.8,top_p=.9,seed='sha256item_id first8hex plus candidate index',
                cases='All20 held-out parameter-variant challenge cases,12 reused natural diagnostics and2user diagnostics. No reranker or reference-based candidate selection.')),
        evaluation='20prospective parameter variants of20 training families +12reused natural mT5diagnostics +2reused usercall inputs +20seeded oldvalidation regressions. Beam output on54;3 sampled drafts on34. Judge useful response, speaker/fact consistency, diversity and own-anchor Singlish meaning separately.',
        limitations='Assistant-authored templated positives/negatives and assistant audit, not independent human gold. Validation/challenge share scenario families with training but parameterized contexts differ. Reused diagnostics are development. One seed and one contrast weight. No locked project test or automatic promotion.',
        research_sources=['https://aclanthology.org/2021.sigdial-1.20/','https://aclanthology.org/2020.acl-main.428/'],
        method_note='Multi-reference SFT plus a bounded sequence-CE margin contrast; not an exact replication of either cited architecture/unlikelihood loss.',
        team_original_sha256=TEAM_SHA,team_data_unchanged=True,default_changed=False,locked_test_read=False,
        payload_manifest={name:sha(path) for name,path in paths.items()})
    save('recipe.json',recipe)
    p=ROOT/'experiments/sealion_creative_v1_recipe.json';p.write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open(OUT/'payload.tgz','w:gz') as tar:
        for name,path in paths.items():tar.add(path,arcname=name,recursive=False)
        tar.add(OUT/'recipe.json',arcname='recipe.json',recursive=False)
    print(json.dumps(dict(counts=recipe['data_counts'],sha256=sha(OUT/'payload.tgz'))))


if __name__=='__main__':main()
