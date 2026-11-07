#!/bin/bash
#SBATCH --job-name=v5-actor-resume
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/3.13.1-GCCcore-13.3.0
source "${PROJECT_ROOT:?set original repository root}/.venv/bin/activate"
export PYTHONPATH="$PWD/src"
export PATH="$HOME/.local/share/replycoach-ollama/bin:$PATH"
export OLLAMA_MODELS="$HOME/.cache/replycoach-ollama/models"
export OLLAMA_MODEL=gemma4:12b OLLAMA_NUM_PARALLEL=1
export OLLAMA_HOST="127.0.0.1:$((20000+SLURM_JOB_ID%30000))"
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
mkdir -p artifacts
python - <<'PY'
import json,hashlib
from pathlib import Path
for f,digest in json.loads(Path('recipe.json').read_text())['files_sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
print('Frozen source, prompts, diagnostic labels and validation hashes verified.',flush=True)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv
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
# Use the installed frozen model; prepare_judge pins its digest. No pull/update.
python -u scripts/run_rl_readiness.py actor_stress --cases data/reward_diagnostics/actor_outputs_v1.jsonl --out artifacts/actor_reward_completed.json --resume-from prior_report.json
ollama ps
