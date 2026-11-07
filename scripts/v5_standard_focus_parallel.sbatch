#!/bin/bash
#SBATCH --job-name=v5-focus-parallel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/3.13.1-GCCcore-13.3.0
source "${PROJECT_ROOT:?}/.venv/bin/activate"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export FOCUS_BASE="$PROJECT_ROOT/artifacts/mt5_base_after_spans_spk/best"
IFS=',' read -r -a focus_devices <<< "${CUDA_VISIBLE_DEVICES:?SLURM must allocate GPUs}"
[ "${#focus_devices[@]}" -eq 2 ]
CUDA_VISIBLE_DEVICES="${focus_devices[0]}" python -u scripts/run_standard_focus_v1.py capacity > capacity.log 2>&1 &
focus_capacity_pid=$!
CUDA_VISIBLE_DEVICES="${focus_devices[1]}" python -u scripts/run_standard_focus_v1.py standard > standard.log 2>&1 &
focus_standard_pid=$!
focus_status=0
wait "$focus_capacity_pid" || focus_status=1
wait "$focus_standard_pid" || focus_status=1
exit "$focus_status"
