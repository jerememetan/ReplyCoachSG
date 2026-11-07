"""Freeze practical example references, complete common coverage, and selection provenance."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'experiments/mt5_reference_pack_v1'


def read(path): return json.loads((ROOT/path).read_text())
def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def write(name, value): (DEST/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    DEST.mkdir(exist_ok=True)
    original_path='artifacts/shared_intent_v1/seeds12/rewrite_s1.json'
    current_path='artifacts/mt5_context_pairs_v1/control_results.json'
    original=read(original_path); current=read(current_path)
    inputs=[json.loads(line) for line in (ROOT/'data/dev/validation.jsonl').read_text().splitlines()]
    shared=current['runs']['seed1_control']['outputs']['final']['beam4']
    oldfinal={r['item_id']:r for r in original['final_validation_outputs']}
    oldbest={r['item_id']:r for r in original['validation_outputs']}
    english={r['item_id']:r for r in shared['english']}; local={r['item_id']:r for r in shared['actual_english']}
    assert current['older_reference']['exact_historical_pairs']==50
    models=dict(
        historical_final=dict(label='Main mT5 example reference: shared-intent seed1 final step784', job=351130,
            checkpoint=original['final_checkpoint'],
            adapter_sha256='d2bb5ea90233de5534f07c9ab7ab6cbf23bf0196388ff6727e3a1bc15673adb9',
            config_sha256='fcd0308d1786cf0bb4f57c63e08d5a2b6549995d77e0dbd655aada5241e8ec13',
            reason='Strongest recorded all-three paired review among three replicated shared-intent rewrite seeds:28/50 versus23/50 and23/50. Exact saved pairs reproduced50/50 in job357554.',
            evidence_limit='Exploratory assistant review on reused validation, not independent human accuracy or a global best across all experiments.',
            training='Older reviewed derivative of190 project conversations:190 English+190 rewritten LOCAL targets+180 original context pairs; one shared rank16 q/v LoRA. Team-authored source files unchanged.'),
        historical_best=dict(label='Alternative mT5 example reference: same seed1 run, epoch4 best-validation', job=351130,
            checkpoint=original['checkpoint'],
            adapter_sha256='3a13c91a7fcb533d60cfb87fafc4283e5bc45b8a0cebfbad3bf3bc337ea9b9ec',
            config_sha256='fcd0308d1786cf0bb4f57c63e08d5a2b6549995d77e0dbd655aada5241e8ec13',
            reason='Earlier snapshot of the same training run; complementary coherent examples and lower reference validation loss. Keep alongside final, without asserting better conversational accuracy.',
            evidence_limit='No equivalent all-three review certifies superiority. This is a different checkpoint of the same model/run, not a new model family.',
            training='Same historical190-conversation shared-intent recipe; selected epoch4,140 optimizer updates.'),
        current_shared=dict(label='Recent mT5 comparison reference: seed1 shared adapter without extra context objective', job=357554,
            checkpoint='/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG_mt5_context_control_v1/artifacts/seed1_control/final',
            resolved_checkpoint='/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG_mt5_context_control_v1/artifacts/seed1_control/step_0784',
            adapter_sha256='46e8940a84e8a6322c78ab2620cf12269216813a27f666d65a7428ae1294a678',
            config_sha256='c369d7527b2ab06b229ab37a871e708f45f20d433fbb66631102745acd7391f7',
            reason='Lowest recent shared-study combined validation CE2.2400; useful current-data comparator. Not established as a conversational upgrade over the historical reference.',
            evidence_limit='Different training targets/counts from the historical models; loss is not a paired quality ranking.',
            training='Current corrected790 English+745 compatible LOCAL rewrites, shared rank16 q/v LoRA,784 updates, no extra context loss.'))
    rows=[]
    for r in inputs:
        key=r['id']; a=oldfinal[key]; b=oldbest[key]
        assert local[key]['standard_anchor']==english[key]['text']
        rows.append(dict(item_id=key,context=[t['text'] for t in r['session_memory']],
                         speakers=[t['speaker'] for t in r['session_memory']],message=r['message']['text'],
                         models=dict(historical_final=dict(english=a['generated_standard'],local=a['generated_local']),
                                     historical_best=dict(english=b['generated_standard'],local=b['generated_local']),
                                     current_shared=dict(english=english[key]['text'],local=local[key]['text']))))
    assert len(rows)==len({r['item_id'] for r in rows})==50
    ids=dict(historical_final=['ITEM-A-030','ITEM-C-023','ITEM-C-038','ITEM-D-038','ITEM-E-025'],
             historical_best=['ITEM-A-030','ITEM-A-054','ITEM-C-045','ITEM-C-055','ITEM-E-007'],
             current_shared=['ITEM-C-001','ITEM-C-038','ITEM-C-045','ITEM-C-055','ITEM-D-038'])
    lookup={r['item_id']:r for r in rows}
    selected={model:[dict(item_id=i,context=lookup[i]['context'],speakers=lookup[i]['speakers'],
                                message=lookup[i]['message'],**lookup[i]['models'][model]) for i in examples]
              for model,examples in ids.items()}
    failures=dict(historical_final=[dict(item_id='ITEM-B-057',reason='Borrower/lender roles are reversed; LOCAL says I borrow my charger.'),
                                   dict(item_id='ITEM-E-028',reason='Accepts meeting after the peer explicitly declines.')],
                  historical_best=[dict(item_id='ITEM-B-057',reason='Borrower/lender roles are reversed.'),
                                   dict(item_id='ITEM-A-038',reason='Invents nine oclock when peer says ten minutes late.')],
                  current_shared=[dict(item_id='ITEM-B-056',reason='Proposes booking a court that is already booked.'),
                                  dict(item_id='ITEM-E-025',reason='Confuses availability for makeup class with checking whether the class is free.')])
    for model, cases in failures.items():
        for case in cases:
            r=lookup[case['item_id']]; case.update(context=r['context'],speakers=r['speakers'],message=r['message'],**r['models'][model])
    random_seed=20261108
    random_rows=random.Random(random_seed).sample(rows,5)
    write('common_validation50.json',rows)
    write('selected_examples.json',dict(selection='Five readability-selected, contextually usable saved pairs per checkpoint. Not random and not accuracy estimates; outputs copied verbatim.',models=selected))
    write('random_comparison5.json',dict(selection='Uniform sample without replacement from common50; same conversations for all checkpoints; outputs unedited.',seed=random_seed,cases=random_rows))
    write('known_failures.json',dict(review='Assistant qualitative notes; plausible alternative intentions are allowed. These are explicit role/fact/context failures.',models=failures))
    manifest=dict(status='references_established', main_reference='historical_final', alternative_reference='historical_best',
        recent_comparator='current_shared', models=models,
        distinction='Two shortlisted historical checkpoints of one mT5-base model/run, plus one recent comparison checkpoint. No universal best or deployment promotion claimed.',
        common_base=read('experiments/mt5_restored_pipeline_v1_source_manifest.json'),
        loader='Use OriginalMT5 compatibility class from src/replycoach/generate/mt5_pretrained_compat.py: preserve separate stored output head despite legacy tie_word_embeddings config. Strict loading must have zero missing/unexpected/mismatched keys.',
        inference='Adapted mT5-base + selected shared adapter; generate STANDARD, then LOCAL_REWRITE using that exact generated English draft and same conversation/speakers. Beam4, no-repeat-ngram3, max48 newtokens, maxsource192, no sampling.',
        target_warning='Historical loss/targets differ from current corrected790 training. Do not compare raw CE across those target sets as accuracy.',
        source_reports_sha256={p:sha(p) for p in [original_path,current_path,'experiments/shared_intent_v1_replication.json','data/dev/validation.jsonl']},
        checkpoint_files_verified_on_cluster='2026-11-08', selected_examples_are_random=False,
        original_team_sha256=sha('data/dev/train.jsonl'), default_changed=False,
        outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.glob('*.json') if p.name!='manifest.json'})
    replay=DEST/'replay_results.json'
    if replay.exists():
        verification=json.loads(replay.read_text())
        assert verification['status']=='complete'
        assert set(verification['models'])==set(models)
        assert all(x['exact_saved_pairs']==50 and not x['differences'] for x in verification['models'].values())
        manifest['current_environment_replay']=dict(job=357652, status='complete', exact_saved_pairs_per_checkpoint=50,
            total_exact_pairs=150, original_checkpoints_unchanged=True, report='replay_results.json', report_sha256=sha('experiments/mt5_reference_pack_v1/replay_results.json'))
    write('manifest.json',manifest)
    lines=['# mT5 example references','','Use the historical final checkpoint as the main mT5 reference, with its earlier epoch4 checkpoint as an alternative. The recent shared adapter is a comparator, not a demonstrated quality upgrade. These are practical example references, not reliable final-model certification.','',
           'The main reference previously passed all three checks on28/50 reused validation conversations in an assistant review, versus23/50 for each of two other shared-intent rewrite seeds. This is not an independent human accuracy score. Its original50 saved English/LOCAL pairs were reproduced exactly in job357554.','',
           'The15 examples below are selected for readability and contextual usefulness. They are not random. See `common_validation50.json` for every shared validation conversation, `random_comparison5.json` for a reproducible uniform sample and `known_failures.json` for explicit failures.','']
    for model, spec in models.items():
        lines.extend(['## '+spec['label'],'',spec['reason'],'',f"Checkpoint: `{spec['checkpoint']}`",'',spec['training'],''])
        for case in selected[model]:
            lines.append('**'+case['item_id']+'**')
            lines.append('')
            lines.extend(f"- {speaker}: {text}" for speaker,text in zip(case['speakers'],case['context'],strict=True))
            lines.extend(['- Incoming: '+case['message'],'- English: '+case['english'],'- Singlish: '+case['local'],''])
        lines.extend(['Known limitations: '+ ' '.join(x['reason'] for x in failures[model]),''])
    lines.extend(['## Reproduce','','Use the common fully adapted base and exact tokenizer specified in `manifest.json`. Load the chosen adapter over that base with the checked OriginalMT5 compatibility loader. This is one shared adapter used for both stages. Generate English first, then rewrite that exact draft to LOCAL using the same conversation and speaker labels. Decoding: beam4, no-repeat-ngram3, maximum48 new tokens, maximum192 source tokens, no sampling.','',
                  'Keep generated-draft and correct-draft diagnostic routes separate. A rewrite can preserve an incorrect English decision. None of these checkpoints changes the project default.'])
    if replay.exists():
        lines.extend(['','All three checkpoints were replayed in job357652: **50/50 exact saved English/Singlish pairs each**,150/150 total. Original checkpoints were unchanged. This checks reproducibility, not correctness.'])
    lines.extend(['','Runnable demo (on the cluster where these paths are available; CUDA required):','',
                  '```bash',
                  'PYTHONPATH=src python scripts/demo_mt5_reference_v1.py \\',
                  '  --reference historical_final \\',
                  '  --inputs experiments/mt5_reference_pack_v1/common_validation50.json \\',
                  '  --output /tmp/mt5_reference_demo.json',
                  '```','',
                  'Choose `historical_best` or `current_shared` for the other references. For your own inputs, supply a JSON list of objects with `item_id`, `context` (text list), `speakers` (matching user/peer/other list), and `message`. The script refuses to overwrite an existing output file.'])
    if (DEST/'training_specs.json').exists():
        lines.extend(['','The complete verified architecture, inherited training stages, per-checkpoint datasets, hyperparameters, metrics and sample provenance are in [training_specs.json](training_specs.json). A and B are snapshots of one training run; C is a separate shared-adapter run.'])
    (DEST/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(main='historical_final',alternative='historical_best',comparator='current_shared',common_cases=len(rows),selected_examples=15,random_ids=[r['item_id'] for r in random_rows])))


if __name__=='__main__': main()
