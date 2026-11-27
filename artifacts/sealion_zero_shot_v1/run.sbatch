#!/bin/bash
#SBATCH --job-name=sealion-zero-shot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --partition=project
#SBATCH --account=cs425
#SBATCH --qos=cs425qos
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
unset PYTHONPATH
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
RCS_PROBE="${HOME}/ReplyCoachSG_sealion_zero_shot_v1_20261108"
RCS_BUNDLE="${HOME}/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1"
cd "${RCS_PROBE}"
"${HOME}/ReplyCoachSG/.venv/bin/python" -I run_sealion_zero_shot_v1.py \
    --bundle "${RCS_BUNDLE}" --inputs inputs.json --expected expected_repair25.json \
    --output results.json
