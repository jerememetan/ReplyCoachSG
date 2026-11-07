#!/bin/bash
#SBATCH --job-name=v5-volume500
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
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export VOLUME_BASE="$PROJECT_ROOT/artifacts/mt5_base_after_spans_spk/best"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u - <<'PY'
import hashlib,json,os
from pathlib import Path
from transformers import AutoTokenizer
from replycoach.generate.finetune_v5 import item_rows,context_pairs
from replycoach.generate.synthetic_v5 import load_synthetic
from replycoach.generate.style_control import mt5_input
r=json.loads(Path('recipe.json').read_text())
for f,digest in r['files_sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
protected=item_rows(Path('data/dev/train.jsonl'))+item_rows(Path('data/dev/validation.jsonl'))
tokenizer=AutoTokenizer.from_pretrained(os.environ['VOLUME_BASE'],local_files_only=True)
counts={}
for count in (190,500):
    rows=load_synthetic(Path(f'reviewed{count}.jsonl'),protected)
    assert len(rows)==2*count
    counts[count]={x['item_id'] for x in rows}
    for row in rows:
        assert len(tokenizer(mt5_input(row['style'],row['message'],row['context'],True,row['speakers']))['input_ids'])<=128,row['item_id']
        assert len(tokenizer(row['response'])['input_ids'])<=48,row['item_id']
assert counts[190]<counts[500]
assert len(item_rows(Path('data/dev/train.jsonl')))==380
assert len(context_pairs(Path('data/dev/train.jsonl')))==180
print('Frozen data/source hashes, nested subsets, protected overlaps, and actual token budgets verified.',flush=True)
PY
for SEED in 425 1 2; do
  for ARM in control8 control_long plus190 plus500; do
    EPOCHS=8
    STEPS=280
    args=()
    case "$ARM" in
      control_long) EPOCHS=23; STEPS=784 ;;
      plus190) STEPS=472; args+=(--synthetic-items "$PWD/reviewed190.jsonl" --synthetic-count 190) ;;
      plus500) STEPS=784; args+=(--synthetic-items "$PWD/reviewed500.jsonl" --synthetic-count 500) ;;
    esac
    python -u -m replycoach.generate.finetune_v5 \
      --base-model "$VOLUME_BASE" --output-dir "$PWD/artifacts" --output-name "${ARM}_s${SEED}" \
      --seed "$SEED" --speaker-labels --context-pairs \
      --epochs "$EPOCHS" --patience 100 --max-steps "$STEPS" --learning-rate 3e-4 --batch-size 16 \
      "${args[@]}"
  done
done
