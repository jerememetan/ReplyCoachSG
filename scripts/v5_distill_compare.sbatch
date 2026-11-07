#!/bin/bash
#SBATCH --job-name=v5-mt5-distill
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
source "${PROJECT_ROOT:?set original repository root}/.venv/bin/activate"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export DISTILL_BASE="$PROJECT_ROOT/artifacts/mt5_base_after_spans_spk/best"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u - <<'PY'
import hashlib,json,os
from pathlib import Path
from transformers import AutoTokenizer
from replycoach.generate.finetune_v5 import item_rows,context_pairs
from replycoach.generate.teacher_v5 import load_teacher_targets
from replycoach.generate.style_control import mt5_input
r=json.loads(Path('recipe.json').read_text())
for f,digest in r['files_sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
original=item_rows(Path('data/dev/train.jsonl'))
teacher,manifest=load_teacher_targets(Path('teacher_reviewed.json'),original,Path('data/dev/train.jsonl'))
assert len(original)==len(teacher)==380
assert len(context_pairs(Path('data/dev/train.jsonl')))==180
tokenizer=AutoTokenizer.from_pretrained(os.environ['DISTILL_BASE'],local_files_only=True)
long_inputs=[]
for row in teacher:
    if len(tokenizer(mt5_input(row['style'],row['message'],row['context'],True,row['speakers']))['input_ids'])>128:
        long_inputs.append((row['item_id'],row['style']))
    assert len(tokenizer(row['response'])['input_ids'])<=48,('Teacher target would truncate',row['item_id'],row['style'])
assert not long_inputs,('Teacher saw context student would truncate',long_inputs)
print('Hashes, audited targets, identical 190 input IDs/contexts/roles, 180 unchanged context pairs and actual tokenizer budgets verified.',flush=True)
PY
for SEED in 425 1 2; do
  for ARM in control teacher mixed; do
    args=()
    case "$ARM" in
      teacher) args+=(--teacher-targets "$PWD/teacher_reviewed.json" --teacher-mode replace) ;;
      mixed) args+=(--teacher-targets "$PWD/teacher_reviewed.json" --teacher-mode augment) ;;
    esac
    python -u -m replycoach.generate.finetune_v5 \
      --base-model "$DISTILL_BASE" --output-dir "$PWD/artifacts" --output-name "${ARM}_s${SEED}" \
      --seed "$SEED" --speaker-labels --context-pairs \
      --epochs 23 --patience 100 --max-steps 784 --learning-rate 3e-4 --batch-size 16 \
      "${args[@]}"
  done
done
