"""Freeze paired-goal pilot sources and evaluation protocol into an isolated bundle."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    files=[*sorted(ROOT.glob('src/**/*.py')),
        *[ROOT/'scripts'/n for n in ['prepare_mt5_act_follow_v1.py','package_mt5_act_follow_v1.py','run_mt5_reply_act_v1.py','run_mt5_act_follow_v1.py','v5_mt5_act_follow_v1.sbatch']],
        ROOT/'tests/test_act_follow_v1.py',ROOT/'data/dev/train.jsonl',ROOT/'data/dev/validation.jsonl',
        *sorted(ROOT.glob('data/training_expansion/mt5_act_follow_v1/*.json')),
        *sorted(ROOT.glob('data/training_preferences/mt5_reply_act_v1/*.json'))]
    recipe=dict(experiment='mt5_act_follow_v1',purpose='Explicitly teach different context-compatible reply acts for the same conversation; distinguish supplied-act control from automatic intent prediction',
        base_checkpoint='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_standard_val_v1/artifacts/full_lr3e4/best',
        rewriter='/common/home/users/h/hm.teoh.2024/ReplyCoachSG_hybrid_inference_v1/checkpoints/local_rewrite',
        seed=425,arms=['direct_control','gated_paired','text_goal_paired'],
        training={'original_conversations':190,'paired_conversations':24,'paired_targets':48,'total_targets':238,'backbone_lr':3e-5,'head_lr':3e-4,'gate_lr':.01,'weight_decay':.01,'batch_size':2,'gradient_accumulation':4,'epochs_max':6,'patience':2,'max_source':192,'max_conditioned_source':256,'max_target':48},
        objectives={'all_arms':'Teacher-forced reply CE on identical 238 targets plus .3 weighted classifier BCE on original examples only; train-only pos weights clipped [1,5]',
            'conditional_arms':'.2 * max(0, .2 + mean-token NLL(target|correct acts) - mean-token NLL(target|counterpart acts)) on new paired examples only. Correct and counterpart acts share the identical conversation. No negative reply strings or RL.',
            'classifier':'Only original training annotations supervise classifier; alternatives are not uniquely inferred intentions.',
            'gated':'Same zero-initialized residual architecture as v2, scale=.1*tanh(gate); hard supplied act vectors in renderer training.',
            'text':'Same mT5 weights encode original conversation for head and re-encode conversation plus explicit ordinary-language reply goal for decoder. Extra pass, no special new tokens.',
            'predicted_inference':'Choose maximum independent-Bernoulli likelihood joint act pattern among observed training-only patterns, then render with that hard act set. No reference labels or reply text used.'},
        validation={'original':50,'paired_conversations':8,'paired_targets':16,'selection':'.5 original supplied-goal validation token CE + .5 paired supplied-goal validation token CE; include epoch0; no evaluation-based selection'},
        evaluation={'fresh_conversations':16,'supplied_goal_outputs_per_arm':32,'automatic_outputs_per_arm':16,'conditioning_disabled_outputs_per_arm':32,'original_validation_automatic_outputs_per_arm':50,
            'primary':'Assistant exploratory review of requested-act compliance AND contextual correctness for both goals of each evaluation conversation. Require both for paired success; changed strings alone are not success. Do not substitute exact reference match for semantic correctness.',
            'secondary':'Automatic predicted route judged as any context-compatible reply. No exact preferred-act target where preferences are unobserved. Review LOCAL meaning preservation separately.',
            'failure_policy':'Blank STANDARD or overbudget rewrite stays explicit failure, no guessed substitutions; no automatic regeneration to hide failures.',
            'scope':'Assistant-authored newly frozen development conversations, structurally related to training. Not human gold or locked product test. Inspect once after completion; any later reuse is development, not fresh evaluation.',
            'stop_rule':'If supplied acts still do not control replies on fresh development cases, do not claim classifier tuning will fix decoder behavior or launch more runs without discussing findings.'},
        limits=['Single seed; small and assistant-authored labels/targets','Conditional arms add ranking objective and supplied information; not matched compute or an isolated interface ablation','Lower privileged validation CE does not establish automatic reply quality','Original validation heavily reused historically','No original test data read and no app defaults changed'],
        files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    p=ROOT/'experiments/mt5_act_follow_v1_recipe.json';p.write_text(json.dumps(recipe,indent=2)+'\n')
    archive=Path('/tmp/replycoach_mt5_act_follow_v1.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)))
        tar.add(p,arcname='recipe.json')
    print('Frozen',len(files),'files to',archive)

if __name__=='__main__':main()
