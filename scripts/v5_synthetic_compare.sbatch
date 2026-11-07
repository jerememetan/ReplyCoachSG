#!/bin/bash
#SBATCH --job-name=v5-synthetic-compare
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/3.13.1-GCCcore-13.3.0
PROJECT_ROOT="${PROJECT_ROOT:?set original repository root}"
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
BASE="$PROJECT_ROOT/artifacts/mt5_base_after_spans_spk/best"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u - <<'PY'
from pathlib import Path
import json,hashlib
from replycoach.generate.finetune_v5 import item_rows
from replycoach.generate.synthetic_v5 import load_synthetic
recipe=json.loads(Path('recipe.json').read_text())
for file,digest in recipe['files_sha256'].items():
    assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
rows=load_synthetic(Path('reviewed.jsonl'),item_rows(Path('data/dev/train.jsonl'))+item_rows(Path('data/dev/validation.jsonl')))
assert len(rows)//2==recipe['synthetic_items']
print('Frozen snapshot hashes and reviewed synthetic data verified',flush=True)
PY
SYN_COUNT="$(python -c 'import json; print(json.load(open("recipe.json"))["synthetic_items"])')"
for SEED in 425 1 2; do
  for ARM in control synthetic; do
    args=()
    if [ "$ARM" = synthetic ]; then
      args+=(--synthetic-items "$PWD/reviewed.jsonl" --synthetic-count "$SYN_COUNT")
    fi
    python -u -m replycoach.generate.finetune_v5 \
      --base-model "$BASE" --output-dir "$PWD/artifacts" --output-name "${ARM}_s${SEED}" \
      --seed "$SEED" --speaker-labels --context-pairs \
      --epochs 8 --patience 2 --max-steps 280 --learning-rate 3e-4 --batch-size 16 \
      "${args[@]}"
  done
done
