#!/bin/bash
#SBATCH --job-name=dec3b-summary
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd /common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_decoder_compare_v1
python3 scripts/summarize_decoder_compare_v1.py
