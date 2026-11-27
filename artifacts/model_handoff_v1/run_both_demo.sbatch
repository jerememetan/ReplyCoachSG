#!/bin/bash
#SBATCH --job-name=replycoach-demo
#SBATCH --nodes=1
#SBATCH --partition=project
#SBATCH --account=cs425
#SBATCH --qos=cs425qos
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=replycoach-demo-%j.log

# sbatch run_both_demo.sbatch [absolute/path/to/your_inputs.json]
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
unset PYTHONPATH
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

RCS_ROOT="${HOME}/ReplyCoachSG_checkpoint_handoff_v1"
RCS_PYTHON="${HOME}/ReplyCoachSG/.venv/bin/python"
RCS_RESULTS="${RCS_ROOT}/demo_outputs/${SLURM_JOB_ID:?Submit this script with sbatch}"
RCS_INPUT_ARGS=(--verify-examples)
if [[ $# -gt 1 ]]; then
    echo 'Usage: sbatch run_both_demo.sbatch [absolute/path/to/your_inputs.json]' >&2
    exit 1
fi
if [[ $# -eq 1 ]]; then
    RCS_INPUT_ARGS=(--inputs "$(realpath -e -- "$1")")
fi
mkdir -p "${RCS_RESULTS}"
RCS_FLAGGED=0

for RCS_NAME in replycoach_mt5_A_v1 replycoach_sealion_repair25_v1; do
    RCS_LABEL=mt5
    if [[ "${RCS_NAME}" == *sealion* ]]; then RCS_LABEL=sealion; fi
    echo "Running ${RCS_LABEL}; results: ${RCS_RESULTS}/${RCS_LABEL}.json"
    RCS_EXIT=0
    "${RCS_PYTHON}" -I "${RCS_ROOT}/${RCS_NAME}/infer.py" \
        --device cuda "${RCS_INPUT_ARGS[@]}" \
        --output "${RCS_RESULTS}/${RCS_LABEL}.json" || RCS_EXIT=$?
    if [[ ${RCS_EXIT} -eq 2 ]]; then
        echo "${RCS_LABEL} saved outputs with status flags or example replay differences; inspect its JSON."
        RCS_FLAGGED=1
    elif [[ ${RCS_EXIT} -ne 0 ]]; then
        exit "${RCS_EXIT}"
    fi
done

"${RCS_PYTHON}" -I - "${RCS_RESULTS}" <<'PY'
import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
outputs = {name: json.loads((root / (name + '.json')).read_text())['outputs']
           for name in ('mt5', 'sealion')}
assert [row['item_id'] for row in outputs['mt5']] == [row['item_id'] for row in outputs['sealion']]
fields = ['item_id'] + [name + '_' + field for name in outputs
                       for field in ('english', 'singlish', 'english_status', 'singlish_status')]
with (root / 'comparison.csv').open('x', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for mt5, sealion in zip(outputs['mt5'], outputs['sealion'], strict=True):
        row = {'item_id': mt5['item_id']}
        for name, reply in [('mt5', mt5), ('sealion', sealion)]:
            row.update({name + '_' + field: reply[field] for field in
                        ('english', 'singlish', 'english_status', 'singlish_status')})
        writer.writerow(row)
print('Replies and comparison saved in:', root)
PY

if [[ ${RCS_FLAGGED} -ne 0 ]]; then exit 2; fi
