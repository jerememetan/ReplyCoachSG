#!/bin/bash
# V5 stage 0 on a SLURM GPU cluster: span corruption on Singapore English chat text. Submit from the repo root:
#   NAME=<run name> sbatch --partition=<partition> --account=<account> --qos=<qos> scripts/v5_spans.sbatch [extra spans_v5 flags]
# MODEL picks the starting model (default mT5-base); anything after the script goes straight to spans_v5:
#   NAME=mt5_base_spans sbatch ... scripts/v5_spans.sbatch
#   NAME=spans_check sbatch ... scripts/v5_spans.sbatch --train-limit 2000 --epochs 1 --eval-every 50
# Then run stage 1 from it: NAME=mt5_base_after_spans MODEL=artifacts/mt5_base_spans/best sbatch ... scripts/v5_pretrain.sbatch
#SBATCH --job-name=v5-spans
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load "${PYTHON_MODULE:?export PYTHON_MODULE to the Python >= 3.10 module the .venv was built with}"
source .venv/bin/activate
# Compute nodes may have no internet; load MODEL once on the login node so it is cached.
export HF_HUB_OFFLINE=1

nvidia-smi --query-gpu=name,memory.total --format=csv
srun --gres=gpu:1 python -m replycoach.generate.spans_v5 \
  --train-text data/public/singlish_train_text.jsonl --validation-text data/public/singlish_validation_text.jsonl \
  --model "${MODEL:-google/mt5-base}" --output-name "${NAME:?set NAME for this run}" "$@"
