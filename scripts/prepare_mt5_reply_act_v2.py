"""Freeze and package a separate reply-act v2 snapshot; does not submit a job."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    recipe=json.loads((ROOT/'experiments/mt5_reply_act_v1_recipe.json').read_text())
    recipe.update(experiment='mt5_reply_act_v2',purpose='Test gentler identity-initialized conditioning and align oracle/predicted training distributions',
        arms=['auxiliary_control','gated_predicted','gated_mixed'],
        condition='Masked mean encoder pool -> sigmoid reply-act posterior -> weighted learned act embeddings -> LayerNorm -> broadcast residual to unmasked existing encoder states. Scale=0.1*tanh(learned gate), gate initialized exactly zero. No new attention positions.',
        train_condition='gated_predicted: predicted soft posterior only. gated_mixed: alternate predicted and training-reference multi-hot routes per batch, parity shifts each epoch. auxiliary_control: classifier trained without conditioning.',
        oracle_limit='gated_mixed trains oracle act vectors on training split only; validation/diagnostic supplied acts remain privileged diagnostic inputs, never deployable predictions. gated_predicted oracle route remains distribution-shifted.',
        gate_lr=.01,residual_bound=.1,gate_initial_value=0.,
        checks='Exact base-memory/loss/generation identity at initialization, gate and embedding gradient flow, padding preservation, reference exclusion, mixed route schedule, blank rewrite handling; actual-checkpoint epoch-zero CE gap asserted <=1e-7 on cluster.',
        inference_routes='none, predicted, oracle for both conditioned arms, none for auxiliary control; disabling learned residual checks whether conditioning actually changes outputs.',
        failure_handling='Blank STANDARD and overbudget rewrite inputs are explicit blocked failures; no guessed substitutions. All eligible routes continue.',
        comparison='Same base checkpoint, original 190/50 development split, provisional act annotations, seed, backbone/head learning rates, 3-epoch maximum, validation predicted CE selection, and unchanged rewriter as v1.',
        evaluation_limits=['Small single-seed development pilot, historically reused validation and inspected diagnostics','Reply acts are provisional assistant annotations, not independently human reviewed or unique valid targets','Uncertainty has two training examples and no validation support','Lower validation loss or a small gate alone does not establish better contextual replies or effective intent control','No locked test access and no app default changes'])
    files=[*sorted(ROOT.glob('src/**/*.py')),
        *[ROOT/'scripts'/n for n in ['prepare_mt5_reply_act_v2.py','run_mt5_reply_act_v1.py','run_mt5_reply_act_v2.py','v5_mt5_reply_act_v2.sbatch']],
        ROOT/'tests/test_reply_act_v1.py',ROOT/'tests/test_reply_act_v2.py',
        ROOT/'data/dev/train.jsonl',ROOT/'data/dev/validation.jsonl',ROOT/'data/evaluation/component_baseline_v1/inputs.json',
        *sorted(ROOT.glob('data/training_preferences/mt5_reply_act_v1/*.json'))]
    recipe['files_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path=ROOT/'experiments/mt5_reply_act_v2_recipe.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
    archive=Path('/tmp/replycoach_mt5_reply_act_v2.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)))
        tar.add(path,arcname='recipe.json')
    print('Frozen',len(files),'files:',archive)

if __name__=='__main__':main()
