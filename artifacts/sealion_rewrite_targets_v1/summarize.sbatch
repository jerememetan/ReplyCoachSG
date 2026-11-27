#!/bin/bash
#SBATCH --job-name=sea-targets-summarize
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
unset PYTHONPATH
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd "${HOME}/ReplyCoachSG_sealion_rewrite_targets_v1"
"${HOME}/ReplyCoachSG_focus_v1/.venv/bin/python" -u summarize_sealion_rewrite_targets_v1.py
