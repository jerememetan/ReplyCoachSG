#!/bin/bash
#SBATCH --job-name=sea-audit-task
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
cd /common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_audited_restart_v1
export PYTHONPATH="$PWD/src:/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1/deps" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python -u scripts/run_sealion_audited_restart_v1.py --stage task
