#!/bin/bash
#SBATCH --job-name=v5-focus-standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:3090:1
#SBATCH --constraint=3090
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/3.13.1-GCCcore-13.3.0
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export FOCUS_BASE="$PWD/artifacts/mt5_base_after_spans_spk/best"
"$PWD/.venv/bin/python" -u scripts/run_standard_focus_v1.py standard
