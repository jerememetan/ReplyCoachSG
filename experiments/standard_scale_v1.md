# STANDARD scaling experiment v1

Submitted 2026-11-04; results pending. Session ends after submission, without waiting for completion or scheduling monitoring.

All three variants start from the same stage1 mT5 checkpoint and train on 790 STANDARD rows: 190 original conversations plus 600 authorized additions (10 individually reviewed, 590 blanket authorized). No LOCAL rows or auxiliary context pairs. Batch size 4, seed 425, 1,584 optimizer updates (8 corpus passes). AdamW, weight decay .01; LoRA q/v rank16, alpha32, dropout .1. Frozen recipe: `standard_scale_v1_recipe.json`.

| Variant | Learning rate | Job | Submission status |
| --- | --- | --- | --- |
| LoRA | 3e-4 | 352975 | Running |
| Full fine-tuning | 1e-4 | 352976 | Pending: group account active-job limit |
| Full fine-tuning | 3e-4 | 352974 | Running |

Each job requests one RTX3090. Scheduler allows one active group job, so the second group job queues automatically. Source/data snapshot checks run inside each worker. Local Python compilation, shell syntax, and 790 unique training IDs checked before submission. Slurm accepted all jobs; running status does not establish training success.

Final checkpoints and reports go to `artifacts/<variant>/` inside the respective remote snapshot directory recorded in `standard_scale_v1_launch.json`. Workers save training generations and reused development STANDARD generations. No locked test read, automatic model promotion, or LOCAL quality claim. Existing default remains unchanged. Next session: inspect exit status and logs, fetch reports, compare development generations and assess LOCAL retention before any promotion.

## Update 2026-11-05

All three Slurm jobs completed with exit code 0 (7m40s, 7m41s, 8m09s respectively for LoRA, full1e-4, full3e-4). Retrieved reports to `artifacts/standard_scale_v1/`. All completed 1584 updates. Training exact matches: LoRA14/790, full1e-4 534/790, full3e-4 730/790. Final training cross entropy: 1.3248, .1641, .0334 respectively. Reused development exact matches: 0/50, 1/50, 0/50; exact wording is not a semantic quality score. Full training clearly fits the corpus more strongly, but improved conversational generalization has not been established. Generated development replies require contextual review and comparison with the baseline. No default change or new training submitted.
