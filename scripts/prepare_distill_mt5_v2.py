"""Freeze train-only teacher-generation pilot and the three-arm student protocol."""
import hashlib
import json
import random
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'distill_mt5_v2'
OUT = ROOT / 'artifacts' / NAME
REMOTE = '/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_' + NAME
PYTHON = '/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python'
DEPS = '/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1/deps'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('A frozen experiment already exists')
    stage = OUT / 'stage'
    old = ROOT / 'artifacts/decoder_compare_v1/stage'
    data = {name: read(old / f'data/{name}.json') for name in (
        'train_task', 'train_rewrite', 'old_validation', 'old_rewrite_validation',
        'evaluation', 'reference_evaluation')}
    for name, rows in data.items():
        save(stage / f'data/{name}.json', rows)
    # Exactly the same 768 conversation inputs; no answer labels in teacher inputs.
    inputs = [{k: r[k] for k in ('item_id', 'context', 'speakers', 'message')} for r in data['train_task']]
    save(stage / 'data/train_inputs.json', inputs)
    if {r['item_id'] for r in inputs} & {r['item_id'] for name in ('old_validation', 'evaluation') for r in data[name]}:
        raise ValueError('Training/evaluation identifiers overlap')
    shutil.copytree(ROOT / 'src/replycoach', stage / 'src/replycoach', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for folder, names in [('scripts', ['run_decoder_compare_v1.py', 'run_distill_mt5_v2.py']),
                          ('tests', ['test_distill_mt5_v2.py'])]:
        (stage / folder).mkdir()
        for name in names:
            shutil.copy2(ROOT / folder / name, stage / folder / name)
    previous = read(old / 'recipe.json')
    specs = dict(sealion=dict(
        root='/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_sealion_audited768_bc_warmup_v1/outputs/C',
        english='423be0f0f9abdfa44554614af44773b0a490a3dd45279cfc9d132c841d5f6bd5',
        singlish='ce7244111fb1873db9940230c34dad4e2ab1d21d8f97c2ef4d96abeebb81156d'),
        qwen=dict(root='/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_qwen_full_sealion_recipe_v1/outputs/qwen',
                  english='52ab46dbeba066927b8ca35f58d45f73236498166606117cf13081af230b0dd5',
                  singlish='5c46532ea0c970670a29f08244487befbaa707b812964927e46ee2924082d342'))
    teachers = {arm: {phase: dict(path=spec['root'] + '/' + ('task' if phase == 'english' else 'rewrite') + '/selected',
                                  sha256=spec[phase], selected_epoch=2) for phase in ('english', 'singlish')}
                for arm, spec in specs.items()}
    pilot_ids = random.Random(425).sample([r['item_id'] for r in inputs], 24)
    student_manifest = read(ROOT / 'experiments/distill_mt5_v2_student_base_manifest.json')
    if student_manifest['files']['model.safetensors']['sha256'] != '20b20e1ac9382c2ca83b6e0b2c665ec4fb1104f853c0e2489d9b0ad205f871ac':
        raise ValueError('Unexpected student base')
    save(stage / 'student_base_manifest.json', student_manifest)
    recipe = dict(experiment=NAME, remote_root=REMOTE, seed=425, models=previous['models'], teachers=teachers,
        method='Response/sequence distillation; standard reply cross-entropy on reviewed teacher text; no token-logit alignment.',
        question='Does reviewed SEA-LION or Qwen supervision improve English and English-conditioned Singlish in the same smaller mT5 student?',
        pilot_ids=pilot_ids, pilot_selection='Uniform sample of 24 of768 training conversations, random.Random(425), before seeing outputs.',
        generation=dict(english_inputs=768, singlish_inputs='Only accepted English outputs',
                        prompt='Existing native teacher prompts; no chat template', beams=4, no_repeat_ngram_size=3,
                        max_new_tokens=64, max_total_tokens=384, temperature=None),
        review=dict(reviewer='Assistant exploratory review; no independent human-gold claim',
                    english='Accept contextually valid selectable replies; reject explicit contradictions, role reversals, missed requests, broken output.',
                    singlish='Preserve accepted English meaning, roles, tense, polarity, conditions, names, times and numbers; natural local style.',
                    edits='Record exact raw-output digest, accept/reject reason, and explicit edited_target for every correction.',
                    original='Use previously audited766 paired targets, exclude two known mismatched pairs.',
                    matching='Common accepted conversation subset across original/SEA-LION/Qwen; same order; report attrition.'),
        student=dict(base=student_manifest['path'], base_parameters=582401280,
                     lineage='mT5-base → historical full-model Singlish span adaptation → full-model speaker-labelled CoSEM/DailyDialog public replies; no task adapter loaded.',
                     precision='FP32 to preserve the historical adapted checkpoint loader and output-head scaling',
                     adapters='Separate fresh English and Singlish rank16 q/v LoRA per arm; same seed and initialization; one frozen shared base.',
                     lora=dict(rank=16, alpha=32, dropout=.1, target_modules=['q','v'], trainable_parameters=1769472),
                     learning_rate=3e-4, epochs=12, microbatch=1, accumulation=8,
                     weight_decay=.01, gradient_clip=1., warmup_fraction=.05, schedule='Cosine to zero',
                     source_limit=512, target_limit=128, generation_new_tokens=64,
                     selection='Common original-label development CE per stage; save every epoch and fixed final epoch; no evaluation-output checkpoint selection.',
                     launch_gate='No student fitting until paired pilot review and finite-gradient/optimization check pass, then complete target review and freeze common subset.'),
        evaluation=dict(development_english=50, development_rewrite=42, reused_pipeline=75, reused_fixed_reference=24,
                        locked_test_used=False, independent_test=False,
                        axes=['English contextual appropriateness', 'English→Singlish meaning fidelity', 'naturalness', 'style fit'],
                        routes=['student English→student Singlish', 'common fixed reference English→student Singlish'],
                        labels='Original control labels include human/synthetic/assistant-revised examples; not pure human gold.'),
        limitations=['Same input pool does not create new conversational coverage; it changes supervision.',
                     'Teacher training already included these inputs, so outputs can reproduce training labels.',
                     'The common mT5 base already has SEA-language adaptation; gains measure added teacher supervision on that base.',
                     'Teacher packages differ beyond pretraining languages; this cannot isolate a causal language-pretraining effect.',
                     'Filtering and assistant edits confound pure raw-teacher imitation; report edits, exclusions and raw targets.',
                     'One student seed initially; development has been repeatedly inspected. No independent final accuracy claim.'],
        source_sha256={str((old / f'data/{name}.json').relative_to(ROOT)): sha(old / f'data/{name}.json') for name in data},
        protected_sha256={p: sha(ROOT / p) for p in ['data/dev/train.jsonl', 'data/dev/validation.jsonl']},
        app_default_changed=False)
    common = ['#!/bin/bash', '#SBATCH --partition=student', '#SBATCH --account=cs701',
              '#SBATCH --qos=studentqos', '#SBATCH --cpus-per-task=4', '#SBATCH --mem=32G',
              '#SBATCH --gres=gpu:1', '#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max',
              '#SBATCH --output=slurm-%x-%j.out', 'set -euo pipefail', 'module purge',
              'module load Python/3.13.1-GCCcore-13.3.0', 'cd ' + REMOTE,
              'export PYTHONPATH="$PWD/src:' + DEPS + '" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false',
              'export HF_HOME="$PWD/hf_cache" TMPDIR="$PWD/tmp"']
    for arm in teachers:
        for phase in ('english', 'singlish'):
            for scope in ('pilot', 'full'):
                name = f'{arm}_{phase}_{scope}'
                lines = common[:1] + ['#SBATCH --job-name=distill-' + name, '#SBATCH --time=04:00:00'] + common[1:]
                lines += [PYTHON + ' -u scripts/run_distill_mt5_v2.py --teacher ' + arm + ' --stage ' + phase + ' --scope ' + scope]
                (stage / (name + '.sbatch')).write_text('\n'.join(lines) + '\n')
    recipe['payload_manifest'] = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    save(stage / 'recipe.json', recipe)
    save(ROOT / 'experiments' / (NAME + '_recipe.json'), recipe)
    with tarfile.open(OUT / 'payload.tgz', 'w:gz') as archive:
        for p in sorted(stage.rglob('*')):
            if p.is_file():
                archive.add(p, arcname=str(p.relative_to(stage)), recursive=False)
    save(OUT / 'package.json', dict(sha256=sha(OUT / 'payload.tgz'), bytes=(OUT / 'payload.tgz').stat().st_size))
    print(json.dumps(dict(remote=REMOTE, pilot_ids=pilot_ids, payload_bytes=(OUT / 'payload.tgz').stat().st_size)))


if __name__ == '__main__':
    main()
