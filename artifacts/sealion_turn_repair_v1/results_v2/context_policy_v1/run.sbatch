#!/bin/bash
#SBATCH --job-name=sea-context-policy
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
unset PYTHONPATH
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "${HOME}/ReplyCoachSG_sealion_turn_repair_v1/context_policy_v1"
"${HOME}/ReplyCoachSG_focus_v1/.venv/bin/python" -I run_sealion_context_policy_v1.py --cached-baselines baseline.json
