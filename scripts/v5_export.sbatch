#!/bin/bash
# V5 development export for the team's judging pipeline: load_v5() answers the 50 validation conversations and
# scripts/export_dev_outputs.py writes experiments/logs/v5_<NAME>_<MEMORY>.jsonl (ignored by Git). From the repo root:
#   NAME=ul025 CHECKPOINT=artifacts/spk1_ul025_s425/best sbatch --partition=<partition> --account=<account> --qos=<qos> scripts/v5_export.sbatch
# MEMORY picks the app's memory setting (M0, M1 or M2; default M2). Send the file to Role E for judging.
#SBATCH --job-name=v5-export
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load "${PYTHON_MODULE:?export PYTHON_MODULE to the Python >= 3.10 module the .venv was built with}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export V5_CHECKPOINT="${CHECKPOINT:?set CHECKPOINT, e.g. artifacts/spk1_ul025_s425/best}"
MEMORY=${MEMORY:-M2}

srun --gres=gpu:1 python scripts/export_dev_outputs.py --generator v5 --mode generator_only --memory "$MEMORY" \
  --output "experiments/logs/v5_${NAME:?set NAME for this export}_${MEMORY}.jsonl" "$@"
