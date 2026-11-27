#!/bin/bash
#SBATCH --job-name=sea-bc-data
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
cd /common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_sealion_large_bc_v1
export PYTHONPATH="$PWD/src:/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1/deps" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python -u scripts/run_sealion_large_bc_v1.py --stage hardware
export PATH="/common/scratch/users/h/hm.teoh.2024/replycoach_public_assets_bc_v1/.local/share/replycoach-ollama/bin:$PATH"
export OLLAMA_MODELS="/common/scratch/users/h/hm.teoh.2024/replycoach_public_assets_bc_v1/.cache/replycoach-ollama/models"
export OLLAMA_HOST="127.0.0.1:$((20000+SLURM_JOB_ID%30000))" OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
ollama serve > ollama-${SLURM_JOB_ID}.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
for i in $(seq 1 90); do if curl -fsS "$OLLAMA_BASE_URL/api/tags" > installed_models.json; then break; fi; sleep 2; done
if /common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python -u scripts/sealion_large_bc_data.py; then
  :
else
  authoring_status=$?
  if [ "$authoring_status" -eq 75 ]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  exit "$authoring_status"
fi
kill "$server_pid"; wait "$server_pid" || true
trap - EXIT
/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python -u scripts/run_sealion_large_bc_v1.py --stage preflight
