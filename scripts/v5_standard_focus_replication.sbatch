#!/bin/bash
#SBATCH --job-name=v5-focus-replicate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/3.13.1-GCCcore-13.3.0
test -x "${PROJECT_ROOT:?}/.venv/bin/python"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export FOCUS_BASE="$PROJECT_ROOT/artifacts/mt5_base_after_spans_spk/best"
"$PROJECT_ROOT/.venv/bin/python" -u scripts/run_standard_focus_replication_v1.py "${FOCUS_SEED:?}"
