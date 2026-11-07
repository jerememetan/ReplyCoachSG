#!/bin/bash
#SBATCH --job-name=onpolicy-sample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:3090:1
#SBATCH --constraint=3090
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/3.13.1-GCCcore-13.3.0
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export ACTOR_BASE="${ACTOR_CHECKPOINT:?}"
"$PROJECT_ROOT/.venv/bin/python" -u scripts/sample_onpolicy_reward_v1.py
