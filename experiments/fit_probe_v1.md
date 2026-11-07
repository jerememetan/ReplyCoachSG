# STANDARD-only training-fit diagnostic — 4 November 2026

User authorized the small-example LoRA/full-finetuning diagnostic after data/duration experiments remained mixed. Purpose: measure whether this training system can learn correct STANDARD responses on its own24 training cases, and whether behavior transfers to newly worded versions of those scenarios. This is a fit/generalization diagnosis, not a candidate product upgrade or estimate of final accuracy.

## Frozen data and comparisons

24 examples from the fresh100 candidates: all10 individually user-approved first-batch examples plus14 additional assistant-checked cases already covered by the user's blanket training authorization. Source identity/approval scope is preserved; do not call all24 human gold. Paraphrases rewrite both histories and incoming messages, preserving essential actors/objects/facts/constraints. They retain underlying scenarios and entities, so success is wording transfer, not proof of general reasoning or24 independent new scenarios. References are not passed to generation prompts. No original held-out test data is read.

Both methods start from the same stage1 speaker-labelled mT5 checkpoint, not a previously fine-tuned model containing the selected examples. Same seed425,24 STANDARD rows, batch4,192/48 untruncated tokens, AdamW weight_decay.01/foreach=False, clipping1.0,400 updates. LoRA uses q/v rank16 alpha32/dropout.1; full fine-tuning updates every model parameter. Two learning rates per method,1e-4 and3e-4, fixed before runs. Both rates are reported. This reduces dependence on one rate but does not establish optimal hyperparameters or perfectly isolate capacity from optimization/dropout differences. Equal updates/examples are not equal FLOPs or optimizer-memory costs.

No LOCAL objective, context-pair augmentation, synthetic subsampling, reward model, DPO or preference labels enter these runs. No early stopping or selection on paraphrases. STANDARD training CE and actual training generations are recorded before training, at100 and400 updates. Paraphrase generation and reference loss are evaluated only after each full400-update run, and never enter the optimizer. Save final model/adapter once per run; full weights are large, so no repeated best-by-loss checkpoint writes. Separate experimental directories preserve paired-style models unchanged.

## Predeclared interpretation

Training adequacy of at least22/24 is an engineering indication that the run learned most of these examples, not a statistical reliability guarantee. Exact-reference match measures memorization/target adherence and is recorded separately; valid alternative responses can pass contextual inspection without matching references. Rate baseline and all four final variants on training cases; rate all four final variants on paraphrases with arm/rate IDs hidden before opening the key. Reviewer authored/checked examples and knows earlier models, so this is not independent human blinding.

If both methods learn training examples but lose adequacy on paraphrases, investigate transfer/data representation. If full succeeds under the tested rates where LoRA does not, restricted adaptation is a candidate explanation requiring controlled follow-up. If neither fits training examples, examine optimization/input construction/targets before scaling. Failure at400 steps under two rates is not proof of impossibility. If both fit and transfer, the issue lies outside this tiny task setting and cannot be attributed to an intrinsic inability to learn.

No default-model promotion or claim about retained LOCAL style follows from this STANDARD-only diagnostic. Source/data hashes, parameter counts, finiteness, update counts, generation coverage and base-checkpoint byte hashes are verified before any result claim.

## Completed diagnostic

Group LoRA job352881 completed in2m41s; personal full-finetuning job352882 completed in2m58s. Each ran both predeclared rates for400 updates, batch4, seed425 on RTX3090. Model, tokenizer and configuration byte hashes match across accounts. Actual trainable parameter counts are1,769,472/584,170,752 for LoRA and582,401,280/582,401,280 for full fine-tuning. Peak allocated CUDA memory was about2.71GB LoRA and10.13GB full. All four final checkpoints and complete reports are retained. Neither method changed product/default models or trained a LOCAL objective.

Both full-finetuning rates reproduced all24 training targets exactly at100 updates and retained24/24 at400. LoRA at100 updates matched0/24 targets at1e-4 and2/24 at3e-4; after400 it matched13/24 and20/24. Normalized exact match ignores case, outer whitespace and terminal punctuation; it is target adherence, not human accuracy. Before training, exact matches were0/24 for all four runs.

| Variant | Final train CE | Training exact /24 | Training adequacy /24 | Paraphrase exact /24 | Paraphrase adequacy /24 |
|---|---:|---:|---:|---:|---:|
| Unchanged stage1 base |2.0000|0|15|0|11|
| LoRA,1e-4 |0.3131|13|17|1|11|
| LoRA,3e-4 |0.0543|20|21|3|15|
| Full,1e-4 |0.000130|24|24|6|18|
| Full,3e-4 |0.001322|24|24|7|18|

The unchanged baseline is the stage1 pretrained checkpoint, **not the selected deployed V5 model**. All methods start from that same baseline. The predeclared22/24 training-adequacy gate is passed by both full runs, not by either LoRA run under these rates/update budgets. This is an engineering diagnostic threshold, not a reliability/significance guarantee. Valid alternative refusals and choices count as adequate even if they do not match targets; uncertainty does not count as pass.

## Interpretation

**The model can learn these STANDARD training targets.** Full fine-tuning achieves exact target fit quickly, so this setup does not show an intrinsic inability of mT5 to learn or a universally broken input/target pipeline. Full also improves wording-transfer adequacy on these scenarios:18/24 versus15/24 for the stronger tested LoRA rate and11/24 for the unchanged stage1 baseline. Both full rates have the same count, with some different errors; do not assume identical behavior.

Restricted adaptation and/or optimization under the tested LoRA configuration is a plausible bottleneck. This does not prove LoRA cannot fit with more steps, different ranks/targets/rates, or another optimizer. The experiment tested only rank16 q/v and two rates for400 updates. Full receives much more trainable capacity and has different adapter-dropout structure; equal updates are not equal compute. Learning targets on24 examples after about66.7 average exposures is not the same task as improving large mixed-data training after8 exposures.

Transfer remains imperfect: even full fine-tuning loses6/24 adequate replies when both context and message are reworded. Both full runs identify Studio4/level2 and answer the paraphrased returned-badge question correctly, whereas LoRA alternatives can omit the studio or answer yes to continued possession despite describing a return. The unchanged baseline places the nets at the explicitly excluded office. Both full runs propose meeting after the six-o'clock shift in the paraphrase probe; LoRA alternatives can garble remain-at-work into leave-work-until-six. However, full can still reverse fan/ruler borrowing roles or leave collection timing unclear. Success is narrower wording transfer on the same entities/scenarios, not general role reasoning.

The study is STANDARD-only for both methods. It does not isolate interference between STANDARD and LOCAL objectives, evaluate natural Singlish, demonstrate retention of broader knowledge, or establish product-quality improvement. Do not replace the paired-style default with a24-example diagnostic checkpoint. A supported next experiment is scaling the STANDARD-only full-finetuning comparison with matched data/evaluation while keeping the existing LOCAL rewrite model separate; that experiment has not been run here.

## Rating and verification evidence

All240 replies (24 train +24 paraphrases × baseline/four variants) have adequacy annotations before opening variant labels.98 exact-reference matches were automatically annotated as matching the prechecked target;142 other outputs received assistant inspection. Identical same-case drafts share recorded judgments and no inconsistent identical-text ratings were found. The assistant selected/checked examples, authored paraphrases, knew fit metrics and can infer variants from outputs; this is not independent human blinding or240 human quality labels. Ten targets had explicit user review and14 assistant checks; all paraphrase expectations are assistant-authored. No numerical accuracy claim applies beyond this diagnostic.

Three integration tests passed for intended parameter updates, STANDARD-only/budget checks and generation recording; the review contract test passed for exact matches, valid alternatives and missing-case rejection (4 tests total). Python compilation and batch-shell syntax passed. Training/response token budgets, source/data hashes, method parameter counts,400 actual updates, baseline byte identity and complete generation IDs were verified. No original held-out test data was read, no reward/DPO labels were made and no product default changed.

Consolidated results and hashes: `fit_probe_v1_results.json`; frozen recipe: `fit_probe_v1_recipe.json`; raw reports, anonymous packet/key, full ratings and logs: `artifacts/fit_probe_v1/`. Group LoRA checkpoints: `~/ReplyCoachSG/experiments/fit_probe_v1/artifacts/lora_lr{1e4,3e4}/final/`. Personal full checkpoints: `~/ReplyCoachSG_fit_probe_v1/artifacts/full_lr{1e4,3e4}/final/`. Full diagnostic model weights are experimental STANDARD-only models. Both queues/checkpoints are checked after completion.
