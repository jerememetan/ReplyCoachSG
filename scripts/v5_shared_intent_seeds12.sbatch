#!/bin/bash
#SBATCH --job-name=v5-shared-seeds12
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
source "${PROJECT_ROOT:?set original repository}/.venv/bin/activate"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export DISTILL_BASE="$PROJECT_ROOT/artifacts/mt5_base_after_spans_spk/best"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u - <<'PY'
import json,hashlib,os
from pathlib import Path
from transformers import AutoTokenizer
from replycoach.generate.finetune_v5 import item_rows
from replycoach.generate.teacher_v5 import load_teacher_targets
from replycoach.generate.rewrite_v5 import rewrite_rows
from replycoach.generate.style_control import mt5_input
for f,digest in json.loads(Path('recipe.json').read_text())['files_sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
original=item_rows(Path('data/dev/train.jsonl'))
paired,_=load_teacher_targets(Path('paired_targets.json'),original,Path('data/dev/train.jsonl'))
tokenizer=AutoTokenizer.from_pretrained(os.environ['DISTILL_BASE'],local_files_only=True)
for row in [*original,*paired,*rewrite_rows(paired)]:
    assert len(tokenizer(mt5_input(row['style'],row['message'],row['context'],True,row['speakers'],standard_reply=row.get('standard_reply')))['input_ids'])<=192,row['item_id']
    assert len(tokenizer(row['response'])['input_ids'])<=48,(row['item_id'],row['style'])
print('Frozen data, audit flags, unchanged 190 inputs and untruncated anchors/targets verified.',flush=True)
PY
for SEED in 1 2; do
for ARM in original aligned rewrite; do
  args=()
  if [ "$ARM" != original ]; then args+=(--teacher-targets "$PWD/paired_targets.json"); fi
  if [ "$ARM" = rewrite ]; then args+=(--rewrite-local); fi
  python -u -m replycoach.generate.finetune_v5 \
    --base-model "$DISTILL_BASE" --output-dir "$PWD/artifacts" --output-name "${ARM}_s${SEED}" \
    --seed "$SEED" --speaker-labels --context-pairs \
    --epochs 23 --patience 100 --max-steps 784 --learning-rate 3e-4 --batch-size 16 \
    --max-source-tokens 192 --save-final "${args[@]}"
done
done
