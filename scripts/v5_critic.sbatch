#!/bin/bash
# Submit from the repo root with the same project/account/QoS as V5 training.
# NAME=critic_pilot sbatch --partition=project --account=cs425 --qos=cs425qos scripts/v5_critic.sbatch
# CRITIC_LIMIT=380 NAME=critic_full ... labels the full sheet after pilot inspection.
# Install Ollama into $HOME/.local/share/replycoach-ollama first (see docs/synthetic_v5.md).
#SBATCH --job-name=v5-critic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load "${PYTHON_MODULE:-Python/3.13.1-GCCcore-13.3.0}"
source .venv/bin/activate
export PATH="${OLLAMA_INSTALL_DIR:-$HOME/.local/share/replycoach-ollama}/bin:$PATH"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.cache/replycoach-ollama/models}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:12b}"
export OLLAMA_NUM_PARALLEL=1
# Bind to loopback inside the allocation; choose a job-specific port.
export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 30000))"
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
mkdir -p "$OLLAMA_MODELS"
command -v ollama
nvidia-smi --query-gpu=name,memory.total --format=csv
ollama serve > "artifacts/ollama-${SLURM_JOB_ID}.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
ready=0
for attempt in {1..60}; do
  if curl --silent --fail "$OLLAMA_BASE_URL/api/tags" > /dev/null; then
    ready=1
    break
  fi
  kill -0 "$server_pid"
  sleep 1
done
if [ "$ready" != 1 ]; then
  echo "Ollama did not become ready; inspect artifacts/ollama-${SLURM_JOB_ID}.log" >&2
  exit 1
fi
ollama pull "$OLLAMA_MODEL"
if [ -n "${VOLUME_TARGET:-}" ]; then
  volume_args=()
  if [ -n "${VOLUME_SEED_FILE:-}" ]; then
    volume_args+=(--seed-reviewed "$VOLUME_SEED_FILE")
  fi
  python -u scripts/generate_synthetic_volume.py \
    --target "$VOLUME_TARGET" --max-candidates "${VOLUME_MAX_CANDIDATES:-800}" \
    --out-dir "${VOLUME_OUT_DIR:?set a fresh corpus directory}" \
    --tokenizer "${VOLUME_TOKENIZER:-$SLURM_SUBMIT_DIR/artifacts/mt5_base_after_spans_spk/best}" \
    --protected-synthetic experiments/synthetic_paired60_v1/reviewed.jsonl \
    "${volume_args[@]}"
elif [ -n "${EVAL_POLICY_INPUT:-}" ]; then
  python scripts/calibrate_evaluation_policy.py "$EVAL_POLICY_INPUT" \
    --rubric "${EVAL_POLICY_RUBRIC:?set frozen evaluator rubric}" \
    --output "artifacts/${NAME:?set a new NAME}.json"
elif [ -n "${PAIRED_REVIEW_INPUT:-}" ]; then
  pair_args=()
  if [ -n "${PAIRED_REVIEW_ORIGINAL:-}" ]; then
    pair_args+=(--original "$PAIRED_REVIEW_ORIGINAL")
  fi
  python scripts/review_synthetic_pairs.py "$PAIRED_REVIEW_INPUT" \
    --output "artifacts/${NAME:?set a new NAME}.json" "${pair_args[@]}"
elif [ -n "${REVIEWED_CALIBRATION_INPUT:-}" ]; then
  python scripts/calibrate_reviewed_critic.py "$REVIEWED_CALIBRATION_INPUT" \
    --output "artifacts/${NAME:?set a new NAME}.json"
elif [ -n "${MODE_COMPARISON_INPUT:-}" ]; then
  comparison_args=()
  if [ -n "${MODE_COMPARISON_BASELINE:-}" ]; then
    comparison_args+=(--baseline "$MODE_COMPARISON_BASELINE")
  fi
  python scripts/compare_critic_modes.py "$MODE_COMPARISON_INPUT" \
    --output "artifacts/${NAME:?set a new NAME}.json" "${comparison_args[@]}"
elif [ -n "${CALIBRATION_INPUT:-}" ]; then
  python scripts/calibrate_critic.py "$CALIBRATION_INPUT" --output "artifacts/${NAME:?set a new NAME}.json"
else
  critic_args=()
  if [ "${CRITIC_SCREEN_SINGLE:-0}" = 1 ]; then
    critic_args+=(--screen-single)
  fi
  python scripts/critic_preferences.py "${PREFS_INPUT:-artifacts/ab_prefs1.json}" \
    --output "artifacts/${NAME:?set a new NAME}.json" --limit "${CRITIC_LIMIT:-20}" "${critic_args[@]}"
fi
# Record residency while the model is still loaded, to verify GPU use.
ollama ps
