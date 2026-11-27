#!/bin/bash
#SBATCH --job-name=sea-abstain-replay3
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
unset PYTHONPATH
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "${HOME}/ReplyCoachSG_sealion_abstention_v3"
mkdir -p custom_replay
"${HOME}/ReplyCoachSG_focus_v1/.venv/bin/python" -I run_sealion_abstention_custom_v3.py --inputs "${HOME}/ReplyCoachSG_sealion_abstention_v2/custom_check/inputs.json" --output "${HOME}/ReplyCoachSG_sealion_abstention_v3/custom_replay/results.json"
