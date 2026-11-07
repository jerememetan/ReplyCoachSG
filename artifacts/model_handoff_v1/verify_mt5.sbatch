#!/bin/bash
#SBATCH --job-name=export-mt5-verify
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=project
#SBATCH --account=cs425
#SBATCH --qos=cs425qos
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
unset PYTHONPATH
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG_checkpoint_handoff_v1"
/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG/.venv/bin/python -I verify_checkpoint_handoff_v1.py --bundle "/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG_checkpoint_handoff_v1/replycoach_mt5_A_v1" --output "/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG_checkpoint_handoff_v1/replycoach_mt5_A_v1_verification.json"
