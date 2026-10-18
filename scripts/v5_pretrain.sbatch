#!/bin/bash
# V5 stage 1 on a SLURM GPU cluster. Submit from the repo root:
#   sbatch --partition=<partition> --account=<account> --qos=<qos> scripts/v5_pretrain.sbatch [smoke|full] [extra train_v5 flags]
# Partition, account and QOS belong to each user (run `myinfo` on the SCIS cluster), so they stay out of this file.
# Experiments are set through the environment, and anything after the mode goes straight to train_v5:
#   NAME=mt5_small_dd DATA=dd sbatch ... scripts/v5_pretrain.sbatch
#   NAME=mt5_base MODEL=google/mt5-base sbatch ... scripts/v5_pretrain.sbatch full --learning-rate 3e-4
# Give every experiment its own NAME: a job resumes whatever checkpoint already sits under that name.
#SBATCH --job-name=v5-pretrain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load "${PYTHON_MODULE:?export PYTHON_MODULE to the Python >= 3.10 module the .venv was built with}"
source .venv/bin/activate
# Compute nodes may have no internet; setup caches MODEL on the login node first.
export HF_HUB_OFFLINE=1

MODE=${1:-full}
if [ $# -gt 0 ]; then shift; fi
PAIRS=data/public
case "${DATA:-both}" in
  both) SOURCES=(dailydialog cosem) ;;
  dd) SOURCES=(dailydialog) ;;  # what V4 was trained on
  *) echo "DATA must be both or dd" >&2; exit 1 ;;
esac
ARGS=(--model "${MODEL:-google/mt5-small}" --train-pairs)
for source in "${SOURCES[@]}"; do ARGS+=("$PAIRS/${source}_train_pairs.jsonl"); done
ARGS+=(--validation-pairs)
for source in "${SOURCES[@]}"; do ARGS+=("$PAIRS/${source}_validation_pairs.jsonl"); done
if [ "$MODE" = smoke ]; then
  ARGS+=(--train-limit 2000 --validation-limit 200 --epochs 1 --eval-every 50 --warmup-steps 20 --output-name mt5_v5_smoke)
else
  NAME=${NAME:-mt5_v5_pretrain}
  # EPOCHS=5 sbatch ... extends a finished run: it resumes and trains up to that total.
  ARGS+=(--epochs "${EPOCHS:-3}" --output-name "$NAME")
  # A resubmitted job continues from the last checkpoint instead of starting again.
  if [ -f "artifacts/$NAME/training_state.pt" ]; then
    ARGS+=(--resume-from "artifacts/$NAME/training_state.pt")
  fi
fi

nvidia-smi --query-gpu=name,memory.total --format=csv
srun --gres=gpu:1 python -m replycoach.generate.train_v5 "${ARGS[@]}" "$@"
