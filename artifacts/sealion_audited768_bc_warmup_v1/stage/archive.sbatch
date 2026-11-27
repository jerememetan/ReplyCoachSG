#!/bin/bash
#SBATCH --job-name=sea768-archive
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd /common/scratch/users/h/hm.teoh.2024/ReplyCoachSG_sealion_audited768_bc_warmup_v1
python3 scripts/archive_sealion_large_bc_v1.py
