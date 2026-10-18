#!/bin/bash
# V5 stage 2 on a SLURM GPU cluster: fine-tune a stage 1 checkpoint on the development items. Submit from the repo root:
#   NAME=<run name> sbatch --partition=<partition> --account=<account> --qos=<qos> scripts/v5_finetune.sbatch [extra finetune_v5 flags]
# BASE picks the stage 1 checkpoint (default mT5-small); anything after the script goes straight to finetune_v5:
#   NAME=mt5_small_lora sbatch ... scripts/v5_finetune.sbatch
#   NAME=mt5_base_lora BASE=artifacts/mt5_base/best sbatch ... scripts/v5_finetune.sbatch
#   NAME=mt5_small_full sbatch ... scripts/v5_finetune.sbatch --lora-rank 0 --learning-rate 1e-4
# Every run needs a new NAME; the trainer refuses to overwrite an earlier one.
#SBATCH --job-name=v5-finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load "${PYTHON_MODULE:?export PYTHON_MODULE to the Python >= 3.10 module the .venv was built with}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

nvidia-smi --query-gpu=name,memory.total --format=csv
srun --gres=gpu:1 python -m replycoach.generate.finetune_v5 \
  --base-model "${BASE:-artifacts/mt5_v5_pretrain/best}" --output-name "${NAME:?set NAME for this run}" "$@"
