#!/bin/bash
#SBATCH --job-name=onpolicy-reward
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:3090:1
#SBATCH --constraint=3090
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/3.13.1-GCCcore-13.3.0
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/share/replycoach-ollama/bin:$PATH"
export OLLAMA_MODELS="$HOME/.cache/replycoach-ollama/models"
export OLLAMA_MODEL=gemma4:12b OLLAMA_NUM_PARALLEL=1
export OLLAMA_HOST="127.0.0.1:$((20000+SLURM_JOB_ID%30000))"
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
mkdir -p artifacts
ollama serve > "artifacts/ollama-${SLURM_JOB_ID}.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
ready=0
for attempt in {1..60}; do
  if curl --silent --fail "$OLLAMA_BASE_URL/api/tags" >/dev/null; then ready=1; break; fi
  kill -0 "$server_pid"
  sleep 1
done
test "$ready" = 1
"${PROJECT_ROOT:?}/.venv/bin/python" -u scripts/score_onpolicy_reward_v1.py
