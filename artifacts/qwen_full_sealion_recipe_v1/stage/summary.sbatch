#!/bin/bash
#SBATCH --job-name=qwenfull-summary
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd /common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_qwen_full_sealion_recipe_v1
python3 scripts/summarize_qwen_full_sealion_recipe_v1.py
