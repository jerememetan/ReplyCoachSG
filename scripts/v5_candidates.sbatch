#!/bin/bash
# V5 sample-and-rerank, GPU step: generate and score candidate replies for the validation items and write
# artifacts/candidates_<NAME>.json. Picking replies from it runs on a laptop (rerank_v5 pick). From the repo root:
#   NAME=<run name> CHECKPOINT=artifacts/base_spans_lora/best sbatch --partition=<partition> --account=<account> --qos=<qos> scripts/v5_candidates.sbatch [--samples 16 --top-p 0.9]
#SBATCH --job-name=v5-candidates
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

srun --gres=gpu:1 python -m replycoach.generate.rerank_v5 candidates \
  --checkpoint "${CHECKPOINT:?set CHECKPOINT, e.g. artifacts/base_spans_lora/best}" --name "${NAME:?set NAME for this run}" "$@"
