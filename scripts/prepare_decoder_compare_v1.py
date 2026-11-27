"""Package a frozen SEA-LION-v1-3B versus Qwen2.5-3B base-model experiment."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'decoder_compare_v1'
REMOTE = '/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_' + NAME
OUT = ROOT / 'artifacts' / NAME
SOURCE = ROOT / 'artifacts/sealion_audited768_bc_warmup_v1/stage'
PYTHON = '/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python'
DEPS = '/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1/deps'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve the prepared experiment')
    stage = OUT / 'stage'
    stage.mkdir(parents=True)
    source_recipe = json.loads((SOURCE / 'recipe.json').read_text())
    inputs = dict(train_task='frozen_data/train_task.json', train_rewrite='frozen_data/train_rewrite.json',
                  old_validation='data/old_validation.json', old_rewrite_validation='data/old_rewrite_validation.json',
                  development='frozen_data/dev_task.json', evaluation='frozen_data/evaluation_all.json',
                  reference_evaluation='frozen_data/evaluation_rewrite.json')
    source_hashes = {}
    for destination, relative in inputs.items():
        path = SOURCE / relative
        digest = sha(path)
        if digest != source_recipe['payload_manifest'][relative]:
            raise ValueError('Frozen source mismatch: ' + relative)
        target = stage / 'data' / (destination + '.json')
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[relative] = digest
    shutil.copytree(ROOT / 'src/replycoach', stage / 'src/replycoach',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (stage / 'scripts').mkdir()
    for name in ['run_decoder_compare_v1.py', 'summarize_decoder_compare_v1.py']:
        shutil.copy2(ROOT / 'scripts' / name, stage / 'scripts' / name)
    (stage / 'tests').mkdir()
    shutil.copy2(ROOT / 'tests/test_decoder_compare_v1.py', stage / 'tests/test_decoder_compare_v1.py')
    recipe = dict(
        experiment=NAME, remote_root=REMOTE, seed=425,
        question='Does a regional SEA-LION base offer practical English/Singlish advantages over a similar-size general multilingual decoder after shared task adaptation?',
        models={
            'sealion': dict(repo='aisingapore/SEA-LION-v1-3B', revision='836890428766e94322a4b7de27fb081a604703a9',
                            path='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion3b_probe_v1/model',
                            pretrained_parameters=3178255360, lora_modules=['Wqkv', 'out_proj', 'up_proj', 'down_proj']),
            'qwen': dict(repo='Qwen/Qwen2.5-3B', revision='3aab1f1954e9cc14eb9509a215f9e5ca08227a9b',
                         path=REMOTE + '/models/qwen', pretrained_parameters=3085938688,
                         lora_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])},
        training=dict(initialization='Fresh LoRA on each original base; no project public adaptation for either model.',
                      epochs=4, task_learning_rate=5e-5, rewrite_learning_rate=1e-5,
                      microbatch=1, accumulation=8, rank=8, alpha=16, adapter_dropout=.05,
                      weight_decay=.01, gradient_clip=1., warmup_fraction=.05, schedule='cosine to zero per optimizer update',
                      precision='BF16 frozen base, FP32 adapters', early_stopping=False,
                      task_updates=768, rewrite_updates=384,
                      task_targets_per_epoch=1536, rewrite_targets_per_epoch=766,
                      checkpoint_selection='Within-model minimum validation reference CE is provisional; also evaluate fixed final epoch4. Read generated validation outputs before quality conclusions.'),
        data=dict(training_conversations=768, rewrite_pairs=766, old_validation=50,
                  old_rewrite_validation=42, assistant_development=20, pipeline_evaluation=75,
                  fixed_reference_evaluation=24, source_hashes=source_hashes,
                  provenance='Previously audited mixed human/synthetic/assistant-revised targets; not all human gold.'),
        decoding=dict(beams=4, no_repeat_ngram_size=3, length_penalty=1., max_new_tokens=64,
                      max_total_tokens=384, cache='Native legacy cache for SEA-LION and DynamicCache for Qwen; common search/stopping rules',
                      prompt='Identical transcript text and labels; no chat template; native tokenizer for each base.'),
        evaluation=dict(automatic_quality_score=False, quality_axes=['contextual appropriateness', 'English-to-Singlish fidelity', 'naturalness', 'style fit'],
                        output_checks=['selected-checkpoint own-English pipeline75', 'selected-checkpoint reference-English rewrite24',
                                       'fixed-epoch4 own-English pipeline75', 'fixed-epoch4 reference-English rewrite24'],
                        selection_cases='Only validation groups; evaluation outputs never choose checkpoints',
                        scope='All evaluation cases are reused development, including 50 old validation cases. No independent test claim.'),
        limitations=[
            'This is a practical model-package comparison, not a causal pretraining-language ablation.',
            'Both models are multilingual; Qwen also supports SEA languages.',
            'Pretraining corpora, compute, architecture, tokenizer and publication date differ.',
            'Common rank and analogous attention/MLP targets do not give identical adapter parameter counts; actual counts are recorded.',
            'Common example/update budgets do not imply equal token or FLOP budgets.',
            'One shared learning-rate recipe and one seed do not establish either model best achievable quality.',
            'Reference CE is not compared across tokenizers as a quality score.',
            'These fresh-base runs differ from the existing SEA-LION public-adapted pipeline; preserve that pipeline separately.',
            'No new teacher generation, distillation, default change or locked-test access in this experiment.',
            'All epoch adapters retained. Optimizer states are not saved, so interruption requires a new explicitly named retry.'
        ], default_changed=False,
    )
    common = ['#!/bin/bash', '#SBATCH --partition=student', '#SBATCH --account=cs701',
              '#SBATCH --qos=studentqos', '#SBATCH --cpus-per-task=4', '#SBATCH --mem=32G',
              '#SBATCH --time=04:00:00', '#SBATCH --gres=gpu:1',
              '#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max', '#SBATCH --output=slurm-%x-%j.out',
              'set -euo pipefail', 'module purge', 'module load Python/3.13.1-GCCcore-13.3.0',
              'cd ' + REMOTE,
              'export PYTHONPATH="$PWD/src:' + DEPS + '" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false',
              'export HF_HOME="$PWD/models/hf_cache" TMPDIR="$PWD/tmp"']
    for arm in ('sealion', 'qwen'):
        lines = common[:1] + ['#SBATCH --job-name=dec3b-' + arm] + common[1:]
        lines += [PYTHON + ' -u scripts/run_decoder_compare_v1.py --stage train --arm ' + arm]
        (stage / (arm + '.sbatch')).write_text('\n'.join(lines) + '\n')
    lines = common[:1] + ['#SBATCH --job-name=dec3b-preflight'] + common[1:]
    lines += [PYTHON + ' -m unittest discover -s tests -p test_decoder_compare_v1.py',
              PYTHON + ' -u scripts/run_decoder_compare_v1.py --stage preflight']
    (stage / 'preflight.sbatch').write_text('\n'.join(lines) + '\n')
    summary = ['#!/bin/bash', '#SBATCH --job-name=dec3b-summary', '#SBATCH --partition=student',
               '#SBATCH --account=cs701', '#SBATCH --qos=studentqos', '#SBATCH --cpus-per-task=1',
               '#SBATCH --mem=2G', '#SBATCH --time=00:10:00', '#SBATCH --output=slurm-%x-%j.out',
               'set -euo pipefail', 'cd ' + REMOTE, 'python3 scripts/summarize_decoder_compare_v1.py']
    (stage / 'summary.sbatch').write_text('\n'.join(summary) + '\n')
    recipe['payload_manifest'] = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    save(stage / 'recipe.json', recipe)
    save(OUT / 'recipe.json', recipe)
    save(ROOT / 'experiments/decoder_compare_v1_recipe.json', recipe)
    with tarfile.open(OUT / 'payload.tgz', 'w:gz') as archive:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(stage)), recursive=False)
    save(OUT / 'package.json', dict(bytes=(OUT / 'payload.tgz').stat().st_size,
                                    sha256=sha(OUT / 'payload.tgz'), files=len(recipe['payload_manifest']) + 1))
    print(json.dumps(dict(stage=str(stage), bytes=(OUT / 'payload.tgz').stat().st_size)))


if __name__ == '__main__':
    main()
