#!/bin/bash
# V5 clean CoSEM, GPU step: score every stage 1 pair with and without its message (see generate/clean_v5.py) and
# write artifacts/scores_<pairs file name>.json. Choosing which pairs to keep (clean_v5 filter) needs no GPU.
# From the repo root:
#   CHECKPOINT=artifacts/mt5_base_cosem_full/best sbatch --partition=<partition> --account=<account> --qos=<qos> scripts/v5_clean.sbatch
# PAIRS defaults to data/public/cosem_full_train_pairs.jsonl.
#SBATCH --job-name=v5-clean
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load "${PYTHON_MODULE:?export PYTHON_MODULE to the Python >= 3.10 module the .venv was built with}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

nvidia-smi --query-gpu=name,memory.total --format=csv
srun --gres=gpu:1 python -m replycoach.generate.clean_v5 score \
  --checkpoint "${CHECKPOINT:?set CHECKPOINT, e.g. artifacts/mt5_base_cosem_full/best}" \
  --pairs "${PAIRS:-data/public/cosem_full_train_pairs.jsonl}" "$@"
