#!/bin/bash
#SBATCH --job-name=sea-user-call
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
unset PYTHONPATH
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
RCS_BUNDLE="${HOME}/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1"
"${HOME}/ReplyCoachSG_focus_v1/.venv/bin/python" -I "${RCS_BUNDLE}/infer.py" --device cuda --inputs "${PWD}/inputs.json" --output "${PWD}/outputs.json"
