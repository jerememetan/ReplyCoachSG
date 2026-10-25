#!/bin/bash
# V5 decoding on a SLURM GPU cluster: re-generate the validation replies of a saved checkpoint with other
# generate settings (no training) and write artifacts/decode_<NAME>.json with reply metrics. From the repo root:
#   NAME=<run name> CHECKPOINT=artifacts/mt5_base_lora/best sbatch --partition=<partition> --account=<account> --qos=<qos> scripts/v5_decode.sbatch [decode flags]
#   NAME=base_lora_beam4 CHECKPOINT=artifacts/mt5_base_lora/best sbatch ... scripts/v5_decode.sbatch
#   NAME=base_lora_enc3 CHECKPOINT=artifacts/mt5_base_lora/best sbatch ... scripts/v5_decode.sbatch --encoder-no-repeat-ngram 3
# RUNS re-decodes several stage 2 runs in this one job, from artifacts/<run>/best, named <run>_<NAME>:
#   NAME=halves RUNS="spk1_joint_s425 spk1_joint_s1 spk1_joint_s2" sbatch ... scripts/v5_decode.sbatch
#SBATCH --job-name=v5-decode
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

if [ -z "${RUNS:-}" ]; then
  srun --gres=gpu:1 python -m replycoach.generate.decode_v5 decode \
    --checkpoint "${CHECKPOINT:?set CHECKPOINT, e.g. artifacts/mt5_base_lora/best}" --name "${NAME:?set NAME for this run}" "$@"
else
  for RUN in $RUNS; do
    srun --gres=gpu:1 python -m replycoach.generate.decode_v5 decode \
      --checkpoint "artifacts/$RUN/best" --name "${RUN}_${NAME:?set NAME for this run}" "$@"
  done
fi
