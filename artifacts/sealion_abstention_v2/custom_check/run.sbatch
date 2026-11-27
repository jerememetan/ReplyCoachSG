#!/bin/bash
#SBATCH --job-name=sea-abstain-v2-demo
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
unset PYTHONPATH
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "${HOME}/ReplyCoachSG_sealion_abstention_v2"
"${HOME}/ReplyCoachSG_focus_v1/.venv/bin/python" -I run_sealion_abstention_custom_v1.py --inputs custom_check/inputs.json --output custom_check/original.json --checkpoint "${HOME}/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1/adapters/english"
"${HOME}/ReplyCoachSG_focus_v1/.venv/bin/python" -I run_sealion_abstention_custom_v1.py --inputs custom_check/inputs.json --output custom_check/v1.json --checkpoint "${HOME}/ReplyCoachSG_sealion_abstention_v1/outputs/selective_abstention/selected"
"${HOME}/ReplyCoachSG_focus_v1/.venv/bin/python" -I run_sealion_abstention_custom_v1.py --inputs custom_check/inputs.json --output custom_check/v2.json --checkpoint "${HOME}/ReplyCoachSG_sealion_abstention_v2/outputs/selective_abstention/selected"
