#!/bin/bash
#SBATCH --job-name=distill-qwen_singlish_full
#SBATCH --time=04:00:00
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
cd /common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_distill_mt5_v2
export PYTHONPATH="$PWD/src:/common/home/users/h/hm.teoh.2024/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1/deps" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/hf_cache" TMPDIR="$PWD/tmp"
/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python -u scripts/run_distill_mt5_v2.py --teacher qwen --stage singlish --scope full
