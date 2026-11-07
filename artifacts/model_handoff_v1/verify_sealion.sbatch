#!/bin/bash
#SBATCH --job-name=export-sealion-verify
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
unset PYTHONPATH
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1"
/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python -I verify_checkpoint_handoff_v1.py --bundle "/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1" --output "/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1_verification.json"
