#!/bin/bash
#SBATCH --job-name=v5-register-compare
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
r=json.loads(Path('recipe.json').read_text())
for file,digest in r['files_sha256'].items():
    assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
formal=[json.loads(l) for l in Path('formal.jsonl').read_text().splitlines()]
chat=[json.loads(l) for l in Path('reviewed.jsonl').read_text().splitlines()]
assert len(formal)==len(chat)==59
for a,b in zip(formal,chat):
    assert a['id']==b['id'] and a['replies']==b['replies']
    assert [t['speaker'] for t in a['session_memory']]==[t['speaker'] for t in b['session_memory']]
rows=load_synthetic(Path('reviewed.jsonl'),item_rows(Path('data/dev/train.jsonl'))+item_rows(Path('data/dev/validation.jsonl')))
assert len(rows)==118
print('59 scenarios and reply targets preserved; hashes and training-only overlaps verified',flush=True)
PY
for SEED in 425 1 2; do
  python -u -m replycoach.generate.finetune_v5 \
    --base-model "$BASE" --output-dir "$PWD/artifacts" --output-name "register_s${SEED}" \
    --seed "$SEED" --speaker-labels --context-pairs \
    --epochs 8 --patience 2 --max-steps 280 --learning-rate 3e-4 --batch-size 16 \
    --synthetic-items "$PWD/reviewed.jsonl" --synthetic-count 59
done
