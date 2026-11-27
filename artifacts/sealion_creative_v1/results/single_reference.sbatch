#!/bin/bash
#SBATCH --job-name=sea-creative-single_reference
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
unset PYTHONPATH
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "${HOME}/ReplyCoachSG_sealion_creative_v1"
"${HOME}/ReplyCoachSG_focus_v1/.venv/bin/python" -u run_sealion_creative_v1.py --arm single_reference
