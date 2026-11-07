# mT5 sequence distillation: same inputs, different targets

The deployment/training target remains mT5 (584,170,752 parameters). Gemma is an offline teacher, not a replacement application model. The teacher does not receive authored reference replies, reviewer notes or validation/test inputs. One shared response intention is expressed in STANDARD and LOCAL in a single teacher call, rather than independently selecting intentions for each style.

## Generation and audit

Teacher generation job 350998 uses the cached local Gemma model in an isolated snapshot with exactly `data/dev/train.jsonl` (190 conversations). Its runtime model digest, prompt, options, source hash and failures are recorded. All candidates begin unapproved. The first ten were inspected while generation continued; a full assistant audit is required before training. This is assistant curation, not human gold. The unreliable reward scorer is not used for automatic approval.

Teacher candidates: remote `~/ReplyCoachSG/experiments/mt5_distill_v1_teacher/artifacts/teacher_candidates.json`. Generator: `scripts/generate_teacher_targets.py`. Frozen teacher recipe: `experiments/mt5_distill_v1_teacher_recipe.json`.

## Controlled student comparison

| Arm | Styled training targets | Unchanged context pairs | Total rows |
|---|---:|---:|---:|
| control | 380 original | 180 | 560 |
| teacher | 380 audited teacher | 180 | 560 |
| mixed | 380 original + 380 audited teacher | 180 | 940 |

All arms use the same 190 input conversations and the same base mT5 weights, rank-16 q/v LoRA, alpha 32, dropout .1, learning rate 3e-4, batch 16, source/target limits 128/48, and seeds 425/1/2. Each has 784 optimizer updates, epochs cap 23 and patience 100. The mixed arm cycles through a larger row set and has fewer epoch-level validation opportunities. Equal update counts do not imply identical token/FLOP counts. Original versus teacher alone is the cleanest target-quality comparison because row counts match.

Teacher loader verifies exact training-source SHA256, complete original ID coverage, unchanged inputs and speaker roles, explicit per-item audit flags, and paired targets. The existing synthetic loader's overlap protections are not weakened: teacher supervision is a separate, explicit path for additional targets on the SAME original inputs. No new synthetic situations enter this experiment. Original context-pair targets remain authored targets in every arm.

The frozen batch preflight checks actual tokenizer lengths before training, so the teacher cannot rely on context that mT5 truncates or provide targets longer than mT5 can learn. Other auxiliary objectives, joint generation and context variants are disabled for teacher arms to isolate target changes.

## Evaluation and limits

Use the same 50 development validation conversations; inspect original-reference cross-entropy and actual reply quality separately, including cross-style meaning. Primary qualitative seed is 425. The validation set has already been reused extensively, so results are developmental, not independent final accuracy. The held-out test remains untouched. Teacher-trained wording can receive higher original-reference loss even if a reply is acceptable; do not select the project direction on loss alone.

This is sequence-level distillation using cross-entropy on teacher-written text, not token-logit matching or RL. The student still generates styles independently; paired training targets do not guarantee paired inference consistency. No promise is made that a 584M student will match a 12B teacher. No default checkpoint replacement is automatic.


## Data audit complete

Teacher generation finished: 190 valid paired candidates, zero generation/shape failures. All 190 source conversations and pairs were inspected by the assistant; 29 pairs received an explicit editorial revision. The original teacher outputs remain at `artifacts/mt5_distill_v1/teacher_candidates.json`; full before/after ledger: `artifacts/mt5_distill_v1/teacher_audit.json`. These are assistant-curated teacher targets, not raw-teacher or human-gold supervision.

Audited corpus: `data/distillation/teacher190_v1_reviewed.json`, SHA256 `ee2136d76ca965c9dfa39c59623f81caa90cb703017354e0c7098b055230514c`. Loader validation passed for 380 paired style rows on the unchanged 190 inputs. Trainer, teacher-loader and existing synthetic-loader tests: 47 passed. Frozen training recipe: `experiments/mt5_distill_v1_recipe.json`. Training submission is not evidence of improved mT5 quality; await completed reports and reply inspection.


Job **351013** observed RUNNING on `aloha`, one RTX 3090 (24 GB VRAM). Source/role/token-length preflight passed before loading mT5. Nine runs submitted. Gemma inference is not part of this training job or the student inference path.


## Completed results

Job 351013 completed in 17m59s, exit zero. All nine runs completed exactly 784 optimizer updates. Source reports were retrieved and expected rows, seed, teacher mode, train/validation item counts and identical validation inputs/references verified. Detailed hashes and learning-curve endpoints: `experiments/mt5_distill_v1_results.json`.

| Arm | Mean best original-reference validation loss |
|---|---:|
| control | 2.4959 |
| teacher | 2.9072 |
| mixed | 2.5375 |

Primary seed 425 assistant comparison against control: teacher 19 wins versus 29 control, 43 ties, eight both bad, one identical; mixed 19 wins versus 17 control, 53 ties, seven both bad, four identical. These are subjective assistant development judgments, not independent human accuracy or evidence that a two-vote margin is meaningful. No quality improvement is established.

Important selection limit: teacher-only best epochs are all 2 (70 updates), despite 784 updates completed. Every arm selected its checkpoint using the original authored validation targets. That selection can disfavor acceptable new teacher wording and may select a teacher student before it has learned those targets. Final-step generated outputs/checkpoints were not retained; this run does not prove that distillation generally fails. Nevertheless the selected teacher-only outputs include garbled repetition such as homework repeated three times and forget to forget, which are actual reply-quality problems beyond lexical-reference mismatch. Mixed supervision remains inconclusive.

The original-only control reproduces prior means to four decimal places; one seed differs by .0001 from a prior run, so do not claim bitwise reproduction. The default mT5 model remains unchanged. Next diagnosis should separate checkpoint selection, target learning and adapter capacity before escalating training complexity. RLHF remains conditional on a reliable reward signal and real human preferences, not these assistant labels alone.


## RLHF implementation considerations

mT5 can remain the actor; RLHF does not require deploying Gemma. Genuine human preferences must be distinguished from current assistant-generated labels and Gemma judgments (AI feedback). Preserve acceptance, refusal and plausible-detail policy, allow ties/both-bad/abstention, validate the reward on separately reviewed conversations, and monitor task quality outside the optimized score. A frozen reference penalty and/or continued supervised objective are candidate safeguards, not guarantees. Reward overoptimization evidence: https://arxiv.org/abs/2210.10760 .

Tooling check: standard Hugging Face TRL GRPOTrainer documents support for causal language models, while mT5 is encoder-decoder. A compatible seq2seq RL implementation must be verified or implemented; the earlier GRPO-style idea is not an off-the-shelf trainer choice for mT5. Source: https://huggingface.co/docs/trl/grpo_trainer . No RLHF/GRPO run has been started and no claim is made that it will solve the observed student-generation errors.
