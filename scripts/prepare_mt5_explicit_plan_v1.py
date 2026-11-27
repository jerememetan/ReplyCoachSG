"""Freeze learned-notes -> conditioned reply -> fixed Singlish comparisons."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from transformers import AutoTokenizer
from replycoach.generate.mt5_explicit_plan_v1 import creative_notes,ledger_notes,diagnostic_notes,planner_source,planned_reply_source,generator_schedule
from replycoach.generate.mt5_matched_v1 import source

OUT=ROOT/'artifacts/mt5_explicit_plan_v1'


def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if p.exists() and p.read_text()!=text:raise FileExistsError(p)
    p.write_text(text);return p


def main():
    team=ROOT/'data/dev/train.jsonl';assert sha(team)=='03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7'
    creative={s:read('artifacts/sealion_creative_v1/data/'+s+'.json') for s in ['train','validation','challenge']}
    facts={s:read('artifacts/mt5_cocktail_v1/data/'+s+'.json') for s in ['train','validation']}
    primary=read('artifacts/mt5_cocktail_v1/data/primary.json')
    old=[r for r in primary if r['style']=='STANDARD'];assert len(old)==790
    oldvalid=[r for r in read('artifacts/mt5_cocktail_v1/data/project_validation.json') if r['style']=='STANDARD'];assert len(oldvalid)==50
    add_notes=lambda rows,fn:[dict(r,gold_notes=fn(r)) for r in rows]
    new={s:add_notes(rows,creative_notes) for s,rows in creative.items()}
    planner_train={'creative':new['train'],'factual':add_notes(facts['train'],ledger_notes)}
    planner_valid={'creative':new['validation'],'factual':add_notes(facts['validation'],ledger_notes)}
    natural=read('artifacts/mt5_cocktail_v1/data/natural_challenge.json');diagnostic=read('artifacts/mt5_cocktail_v1/data/diagnostics.json')
    natural=[dict(r,gold_notes=diagnostic_notes(r),scope='reused_natural_development') for r in natural]
    diagnostic=[dict(r,gold_notes=diagnostic_notes(r)) for r in diagnostic]
    assert all(r['gold_notes'] for r in natural+diagnostic)
    evaluation=[dict(r,scope='parameter_heldout_creative') for r in new['challenge']]+natural+diagnostic+[dict(r,scope='reused_project_validation') for r in oldvalid]
    key=lambda r:json.dumps([r['context'],r['speakers'],r['message']],sort_keys=True)
    train_keys={key(r) for r in old+new['train']+facts['train']}
    assert not train_keys&{key(r) for r in new['validation']+facts['validation']+evaluation}
    tok=AutoTokenizer.from_pretrained('/tmp/replycoach_mt5_adapted_tokenizer',local_files_only=True)
    budgets={}
    for name,rows in [(f'planner_{split}_{part}',group) for split,groups in [('train',planner_train),('validation',planner_valid)] for part,group in groups.items()]:
        encoded=[(tok(planner_source(r))['input_ids'],tok(r['gold_notes'])['input_ids']) for r in rows]
        assert all(len(x)<=256 and len(y)<=160 for x,y in encoded),name
        budgets[name]=dict(count=len(rows),max_source=max(len(x) for x,y in encoded),max_target=max(len(y) for x,y in encoded))
    for row in sum(new.values(),[])+natural+diagnostic:
        assert len(tok(planned_reply_source(row,row['gold_notes']))['input_ids'])<=384
        for alt in row.get('alternatives',[]):assert len(tok(alt['english'])['input_ids'])<=48
    for row in old+oldvalid:
        assert len(tok(planned_reply_source(row,''))['input_ids'])+160<=384
        assert len(tok(row['response'])['input_ids'])<=48
    mapping={
        'data/planner_train.json':save('data/planner_train.json',planner_train),
        'data/planner_validation.json':save('data/planner_validation.json',planner_valid),
        'data/ordinary.json':save('data/ordinary.json',old),
        'data/ordinary_validation.json':save('data/ordinary_validation.json',oldvalid),
        'data/new_train.json':save('data/new_train.json',new['train']),
        'data/new_validation.json':save('data/new_validation.json',new['validation']),
        'data/evaluation.json':save('data/evaluation.json',evaluation),
        'schedule.json':save('schedule.json',generator_schedule(790)),
        'source_manifest.json':ROOT/'experiments/mt5_restored_pipeline_v1_source_manifest.json',
        'protected_team.jsonl':team,
    }
    files=list(ROOT.glob('src/**/*.py'))+[ROOT/p for p in [
        'scripts/run_mt5_explicit_plan_v1.py','scripts/prepare_mt5_explicit_plan_v1.py','scripts/summarize_mt5_explicit_plan_v1.py',
        'scripts/run_mt5_adapter_switch_v1.py','tests/test_mt5_explicit_plan_v1.py']]
    mapping.update({str(p.relative_to(ROOT)):p for p in files})
    recipe=dict(experiment='mt5_explicit_plan_v1',account='hm.teoh.2024',seed=4601,
        goal='Conversation -> freely generated facts/roles/constraints/open-choice notes -> explicitly conditioned English reply -> fixed own-English Singlish rewrite. Creative proposals allowed when private intent is absent.',
        base_personal='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_mt5_mlp_generation_v1/model',
        reply_initial_adapter='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_mt5_cocktail_v1/outputs/baseline/step_0784',
        frozen_rewriter='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_mt5_cocktail_v1/outputs/baseline/step_0784',
        reply_initial_sha256=read('artifacts/mt5_cocktail_v1/summary.json')['reports']['baseline']['history'][-1]['sha256'],
        topology='Separate rank16q/v LoRA planner and reply adapters on the same preserved582401280parameter adapted mT5-base;1769472trainable per phase. Reply arms reset same completed cocktail baseline; Singlish uses its frozen shared adapter.',
        planner=dict(train_creative=100,train_factual=800,validation_creative=20,validation_factual=160,
            initialization='Fresh rank16q/v LoRA on frozen adapted base',updates=450,batch=8,creative_per_step=4,factual_per_step=4,
            learning_rate=3e-4,snapshots=[0,150,300,450],selection='Minimum mean of creative/factual validation tokenCE including0; no generation/response challenge selection.',
            label_provenance='Assistant-authored source-context annotations for20families plus factual evidence ledgers. No reference reply is read by gold-note annotators.',max_source=256,max_target=160),
        responders=dict(arms=['no_notes_control','explicit_notes'],updates=300,old_per_step=6,new_per_step=2,
            old_examples=790,new_contexts=100,new_alternatives=300,new_visits_per_context=6,each_alternative_visits=2,
            learning_rate=3e-5,snapshots=[0,75,150,300],
            control='Identical planned-reply input format with empty notes.',
            explicit='Ordinary790 use actual predicted notes. New100 alternate gold and actual in-sample predicted notes, selected by the frozen schedule. Validation and normal inference always use actual predicted notes.',
            selection='Minimum0.5old50tokenCE+0.5new20mean3referenceCE with actual predicted notes for explicit arm, empty notes forcontrol; step0eligible. No oracle/challenge selection.'),
        evaluation=dict(cases=84,creative_parameter_variants=20,reused_natural=12,reused_user=2,reused_old_validation=50,
            routes=['unchanged_baseline_STANDARD','continued_no_notes_control','explicit_actual_predicted_notes','same_explicit_checkpoint_no_notes','same_explicit_checkpoint_gold_notes_diagnostic'],
            oracle_available_cases=34,oracle_policy='Gold notes only on34labelled diagnostics and explicitly marked oracle; never substituted into normal predictions. No oracle for old50 without annotations.',
            decoding=dict(num_beams=4,no_repeat_ngram_size=3,do_sample=False,max_new_tokens=48),
            notes_errors='Record planner string/status/input provenance, then pass exactly that string onward. Never silently repair/replace generated notes. Blank notes remain blank and flagged; no refusal gate.'),
        budget=budgets,weight_decay=.01,clip_norm=1.,precision='FP32',max_reply_source=384,
        limitations='One seed, assistant labels, shared template families, heavily reused development inputs. Planner training predictions are in-sample; gold/predicted mixture is not cross-fitted. Gold-note diagnostics differ in input distribution and are not guaranteed upper bounds. Longer inputs and extra planner inference cost are not matched FLOPs. Lower CE does not prove useful replies.',
        team_original_sha256=sha(team),locked_test_read=False,default_changed=False,
        files_sha256={name:sha(path) for name,path in mapping.items()})
    save('recipe.json',recipe);(ROOT/'experiments/mt5_explicit_plan_v1_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    with tarfile.open(OUT/'payload.tgz','w:gz') as tar:
        for name,p in mapping.items():tar.add(p,arcname=name,recursive=False)
        tar.add(OUT/'recipe.json',arcname='recipe.json',recursive=False)
    print(json.dumps(dict(planner_examples=900,reply_old=790,reply_new=100,evaluation=84,budget=budgets,payload_sha256=sha(OUT/'payload.tgz'))))


if __name__=='__main__':main()
