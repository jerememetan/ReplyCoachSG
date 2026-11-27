#!/bin/bash
#SBATCH --job-name=sealion-style-check
#SBATCH --partition=student
#SBATCH --account=cs701
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module purge
module load Python/3.13.1-GCCcore-13.3.0
cd "${HOME}/ReplyCoachSG_sealion_style_v1"
export PYTHONPATH="$PWD:${HOME}/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1/src:${HOME}/ReplyCoachSG_checkpoint_handoff_v1/replycoach_sealion_repair25_v1/deps"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
RCS_PYTHON="${HOME}/ReplyCoachSG_focus_v1/.venv/bin/python"
"${RCS_PYTHON}" - <<'PY'
import importlib.util
import pathlib
import unittest
suite=unittest.TestSuite()
for path in [pathlib.Path('tests/test_sealion_style_v1.py'),pathlib.Path('tests/test_sealion_beam_v1.py')]:
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name,value in vars(module).items():
        if name.startswith('test_') and callable(value):
            suite.addTest(unittest.FunctionTestCase(value,description=name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
"${RCS_PYTHON}" check_sealion_style_v1.py
