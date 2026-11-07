#!/bin/bash
#SBATCH --job-name=mt5-random10-all3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=project
#SBATCH --account=cs425
#SBATCH --qos=cs425qos
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG_mt5_reference_pack_v1"
module purge
module load Python/3.13.1-GCCcore-13.3.0
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG/.venv/bin/python -u scripts/demo_mt5_reference_v1.py --reference all --inputs "/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG_mt5_reference_pack_v1/random10_1632249497.json" --output "/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG_mt5_reference_pack_v1/artifacts/random10_1632249497_all3.json"
