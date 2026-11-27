"""Read-only per-reference loss audit and intermediate single-reference replies."""
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_sealion_turn_repair_v1 as core

STUDY=Path.home()/'ReplyCoachSG_sealion_creative_v1'


def read(path):return json.loads(Path(path).read_text())


@core.torch.no_grad()
def metrics(model,tok,ordinary,rows):
    model.eval();total=0.;tokens=0
    for row in ordinary:
        e=core.encode(row,tok);n=sum(v!=-100 for v in e['labels'][1:]);total+=float(core.loss(model,e))*n;tokens+=n
    reference_losses=[];negative_losses=[]
    for row in rows:
        reference_losses.append([float(core.loss(model,core.encode(dict(row,positive=a['english']),tok))) for a in row['alternatives']])
        negative_losses.append(float(core.loss(model,core.encode(row,tok,'negative'))))
    per_ref=[sum(v[i] for v in reference_losses)/len(rows) for i in range(3)]
    return dict(ordinary_token_ce=total/tokens,per_reference_mean_ce=per_ref,
        mean_reference_ce=sum(per_ref)/3,
        best_reference_mean_ce=sum(min(v) for v in reference_losses)/len(rows),
        any_reference_better_than_negative=sum(min(v)<n for v,n in zip(reference_losses,negative_losses,strict=True)),
        cases=len(rows),per_case_reference_ce=[dict(item_id=r['item_id'],reference_ce=v,negative_ce=n) for r,v,n in zip(rows,reference_losses,negative_losses,strict=True)],
        note='Post-hoc teacher-forced diagnostics, not generated quality or independent test accuracy. Best-reference CE favours any one supplied reference; not a replacement selection rule.')


@core.torch.no_grad()
def paired(model,tok,cases,path):
    core.activate(model,'default');outputs=[]
    for row in cases:
        out=core.decode(model,tok,row)
        outputs.append(dict(item_id=row['item_id'],context=row['context'],speakers=row['speakers'],message=row['message'],scope=row.get('scope'),
            english=out['reply'],english_status=out['status']))
    core.activate(model,'singlish')
    for row,out in zip(cases,outputs,strict=True):
        sg=core.decode(model,tok,row,out['english']) if out['english'].strip() else dict(reply='',status='blocked_empty_anchor')
        out.update(singlish=sg['reply'],singlish_status=sg['status'],standard_anchor=out['english'])
    report=dict(status='complete',outputs=outputs,training_performed=False,
        flags=[r['item_id'] for r in outputs if r['english_status']!='complete' or r['singlish_status']!='complete'])
    core.save(path,report);return report


def main():
    assert Path.home().name=='hm.teoh.2024'
    core.torch.set_num_threads(4)
    assert not (ROOT/'report.json').exists()
    for name,h in read(ROOT/'payload_manifest.json').items():assert core.sha(ROOT/name)==h,name
    # Exercise the original saved portable loader/examples, without training.
    replay=ROOT/'original_export_replay.json'
    result=subprocess.run([sys.executable,'-I',str(core.BUNDLE/'infer.py'),'--device','cuda','--verify-examples','--output',str(replay)])
    assert result.returncode==0,'Original inference replay failed; preserve log/report for investigation.'
    original=read(replay);assert original['exact_example_pairs']==10 and not original['replay_differences'] and not original['flags']
    training=read(STUDY/'outputs/single_reference/training.json')
    ordinary=read(STUDY/'data/legacy.json');validation=read(STUDY/'data/validation.json')
    recipe=read(STUDY/'recipe.json');cases=read(STUDY/'data/challenge.json')+read(STUDY/'data/natural.json')+read(STUDY/'data/diagnostics.json')
    import random
    cases+=[dict(r,scope='reused_old_validation') for r in random.Random(recipe['seed']).sample(ordinary,20)]
    assert [r['item_id'] for r in cases]==[r['item_id'] for r in read(STUDY/'outputs/original/evaluation.json')['outputs']]
    inventory=read(STUDY/'checkpoint_inventory.json')
    candidates=[('original',core.BUNDLE/'adapters/english',recipe['source_english_sha256'])]
    for arm,step in [('single_reference',75),('single_reference',150),('single_reference',300),('multiple_references',300),('multiple_plus_contrast',300)]:
        entry=next(r for r in inventory if r['arm']==arm and r['step']==step)
        candidates.append((arm+'_'+str(step),Path(entry['path']),entry['sha256']))
    tok=core.checkpoint_tokenizer(core.BUNDLE/'base')
    report=dict(status='running',training_performed=False,source_study=str(STUDY),metrics={},
        original_export_replay=dict(exact_pairs=10,report=str(replay),gpu=original['gpu']),
        default_changed=False,selection_rule_changed=False,
        note='All contexts have already been inspected during development. This investigates reference-coverage loss versus usable drafts; no independent accuracy or automatic promotion.')
    core.save('report.json',report)
    for name,path,h in candidates:
        assert core.sha(path/'adapter_model.safetensors')==h
        model=core.load(path)
        scores=metrics(model,tok,ordinary,validation)
        report['metrics'][name]=dict(checkpoint=str(path),sha256=h,**scores)
        if name in {'single_reference_75','single_reference_150'}:
            model.load_adapter(core.BUNDLE/'adapters/singlish',adapter_name='singlish',is_trainable=False)
            paired(model,tok,cases,'outputs/'+name+'.json')
        assert core.sha(path/'adapter_model.safetensors')==h
        core.save('report.json',report)
        print(name,json.dumps({k:v for k,v in scores.items() if k!='per_case_reference_ce'}),flush=True)
        del model;gc.collect();core.torch.cuda.empty_cache()
    assert core.sha(core.BUNDLE/'adapters/english/adapter_model.safetensors')==recipe['source_english_sha256']
    assert core.sha(core.BUNDLE/'adapters/singlish/adapter_model.safetensors')==recipe['frozen_singlish_sha256']
    report.update(status='complete',source_weights_unchanged=True)
    core.save('report.json',report)
    print('READ-ONLY REVIEW COMPLETE',flush=True)


if __name__=='__main__':main()
