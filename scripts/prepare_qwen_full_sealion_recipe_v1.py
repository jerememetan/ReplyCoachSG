"""Freeze the full public→task→rewrite Qwen replication without replacing models."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'qwen_full_sealion_recipe_v1'
OUT = ROOT / 'artifacts' / NAME
REMOTE = '/common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_' + NAME
PYTHON = '/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python'
DEPS = '/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1/deps'
PUBLIC_ROOT = '/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_full_v1/data/public/sealion_full_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    if OUT.exists():
        raise FileExistsError('Prepared full-pipeline run already exists')
    stage = OUT / 'stage'
    stage.mkdir(parents=True)
    old = ROOT / 'artifacts/decoder_compare_v1/stage'
    previous = json.loads((old / 'recipe.json').read_text())
    shutil.copytree(old / 'data', stage / 'data')
    for p in (stage / 'data').glob('*.json'):
        if sha(p) != previous['payload_manifest']['data/' + p.name]:
            raise ValueError('Task source changed')
    shutil.copytree(ROOT / 'src/replycoach', stage / 'src/replycoach',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (stage / 'scripts').mkdir()
    for name in ['run_decoder_compare_v1.py', 'run_qwen_full_sealion_recipe_v1.py',
                 'summarize_qwen_full_sealion_recipe_v1.py']:
        shutil.copy2(ROOT / 'scripts' / name, stage / 'scripts' / name)
    (stage / 'tests').mkdir()
    for name in ['test_decoder_compare_v1.py', 'test_qwen_public_v1.py']:
        shutil.copy2(ROOT / 'tests' / name, stage / 'tests' / name)
    download = json.loads((ROOT / 'artifacts/decoder_compare_v1/results/models/download_manifest.json').read_text())
    base_path = previous['models']['qwen']['path']
    download['files'] = {path: digest for path, digest in download['files'].items() if path.startswith(base_path + '/')}
    save(stage / 'models/download_manifest.json', download)
    recipe = dict(previous)
    recipe.update(experiment=NAME, remote_root=REMOTE, models={'qwen': previous['models']['qwen']},
                  question='Replicate the historical SEA-LION public-data pass and audited768 C pipeline on Qwen2.5-3B.',
                  reference_experiment='sealion_audited768_bc_warmup_v1/C',
                  preserved_comparison='decoder_compare_v1',
                  public=dict(
                      sources={
                          'train': dict(path=PUBLIC_ROOT + '/train.jsonl', rows=117634,
                                        sha256='93f31a67bb288ee58605e1f0b266ff22f25126693e698e39c1c9adce6dff6d86'),
                          'validation': dict(path=PUBLIC_ROOT + '/validation.jsonl', rows=400,
                                             sha256='e2d604b0002f77a3568ef4e366df66abedd3748727e0096b07a01d2b803e8259')},
                      corpora=dict(cosem=53691, dailydialog=63943), full_passes=1,
                      max_total_tokens=640,
                      token_budget_note='Qwen encodes one of117634 identical public examples as592tokens; public limit640 avoids truncation. Task/rewrite limit remains384.',
                      learning_rate=5e-5, accumulation=16, optimizer_updates=7353,
                      schedule='Constant, matching historical SEA-LION public stage',
                      public_order_seed=1425, checkpoint_every_examples=4000,
                      validation_every_examples=10000, selection='Final full pass, not minimum validation loss',
                      privacy_scope='Previously filtered public exports with historical exclusions; no new generation or independent privacy certification.'),
                  request_interpretation='Full SEA-LION-style pipeline recommended in scope clarification; default announced after collecting the already-completed matched768 comparison.')
    recipe['training'] = dict(previous['training'], epochs=8,
                             initialization='Fresh Qwen LoRA → full public pass → task → copied Singlish rewrite adapter',
                             initial_task_adapter='outputs/public/final_adapter',
                             task_updates=1536, rewrite_updates=768,
                             checkpoint_selection='Provisional within-model reference CE; also export fixed epoch8. Generated-reply review required before promotion.')
    recipe['evaluation'] = dict(previous['evaluation'])
    recipe['evaluation']['output_checks'] = [text.replace('epoch4', 'epoch8') for text in recipe['evaluation']['output_checks']]
    recipe['limitations'] = [text for text in previous['limitations'] if 'fresh-base runs differ' not in text and 'Optimizer states are not saved' not in text]
    recipe['limitations'] += [
        'Full-pipeline and completed task-only Qwen differ in public exposure, epoch budget and cosine schedule length; this is not an isolated public-data ablation.',
        'Replicates the historical SEA-LION C learning-rate recipe, not both B/C branches or the later1291-conversation mixture.',
        'Public source data and Qwen base weights are shared read-only dependencies; preserve their recorded paths.',
        'Public stage saves exact optimizer/RNG resume state; task/rewrite save all epoch adapters but not optimizer states.',
        'The public CoSEM/DailyDialog source terms and original licence records remain applicable; this run does not establish redistribution approval.'
    ]
    common = ['#!/bin/bash', '#SBATCH --partition=student', '#SBATCH --account=cs701',
              '#SBATCH --qos=studentqos', '#SBATCH --cpus-per-task=4', '#SBATCH --mem=32G',
              '#SBATCH --gres=gpu:1', '#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max',
              '#SBATCH --output=slurm-%x-%j.out', 'set -euo pipefail', 'module purge',
              'module load Python/3.13.1-GCCcore-13.3.0', 'cd ' + REMOTE,
              'export PYTHONPATH="$PWD/src:' + DEPS + '" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false',
              'export HF_HOME="$PWD/hf_cache" TMPDIR="$PWD/tmp"']
    for name, hours in [('preflight', 2), ('public', 12), ('pipeline', 8)]:
        lines = common[:1] + ['#SBATCH --job-name=qwenfull-' + name, '#SBATCH --time=%02d:00:00' % hours] + common[1:]
        if name == 'preflight':
            lines += [PYTHON + ' -m unittest discover -s tests -p test_decoder_compare_v1.py',
                      PYTHON + ' -m unittest discover -s tests -p test_qwen_public_v1.py']
        lines += [PYTHON + ' -u scripts/run_qwen_full_sealion_recipe_v1.py --stage ' + name]
        (stage / (name + '.sbatch')).write_text('\n'.join(lines) + '\n')
    lines = ['#!/bin/bash', '#SBATCH --job-name=qwenfull-summary', '#SBATCH --partition=student',
             '#SBATCH --account=cs701', '#SBATCH --qos=studentqos', '#SBATCH --cpus-per-task=1',
             '#SBATCH --mem=2G', '#SBATCH --time=00:10:00', '#SBATCH --output=slurm-%x-%j.out',
             'set -euo pipefail', 'cd ' + REMOTE, 'python3 scripts/summarize_qwen_full_sealion_recipe_v1.py']
    (stage / 'summary.sbatch').write_text('\n'.join(lines) + '\n')
    recipe['payload_manifest'] = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    save(stage / 'recipe.json', recipe)
    save(OUT / 'recipe.json', recipe)
    save(ROOT / 'experiments' / (NAME + '_recipe.json'), recipe)
    with tarfile.open(OUT / 'payload.tgz', 'w:gz') as archive:
        for p in sorted(stage.rglob('*')):
            if p.is_file():
                archive.add(p, arcname=str(p.relative_to(stage)), recursive=False)
    package = dict(bytes=(OUT / 'payload.tgz').stat().st_size, sha256=sha(OUT / 'payload.tgz'),
                   files=len(recipe['payload_manifest']) + 1)
    save(OUT / 'package.json', package)
    print(json.dumps(package))


if __name__ == '__main__':
    main()
