## Current transformer continuation handoff

Read [docs/transformer_resume.md](../../docs/transformer_resume.md) first. Updated 6 November 2026. It supersedes historical pending statuses and next-step suggestions below. Active direction: compare mT5 with Qwen4B/Gemma12B, direct versus explicit planning, and separate incoming-intent/reply-act conditioning before more training. Qwen355488 completed24 cases with no failed stages. Gemma355489 last running; intent355498 last pending. mT5 attempt355490 failed on missing response field; fixed with empty inference targets and retry355500 completed24 cases, both rewrite routes verified. Five tests passed. No default change. Check live accounting before reporting results.

## Historical experiment entries

## Small faithful rewrite diagnostic v3 launched

Job 355463 verified running. See experiments/rewrite_diagnostic_v3.md and its launch/recipe. 24 train/12 author-heldout original LOCAL pairs, 12 meaning probes. Intentional 300-step small-set capacity test, not production model selection. Matched development audit includes incumbent controls. Defaults unchanged.

## Native-target controlled rewrite trials launched

Jobs355353 full_context,355354 anchor_only,355355 exemplar.224train/59component-val native LOCAL targets, all LOCAL text unchanged;183 original+100 CoSEM retained,7+50 quarantined. English normalizations assistant audited/corrected, not human gold.40 new fixed epoch1 STANDARD anchors for generation only. Full mT5 stage1 base,lr3e-5,max8epochs,half-epoch train/val tracking,baseline candidate,patience3. Six tests/preflight passed. Two running/one queued at verification. See experiments/native_rewrite_v2.md/recipe/launch.13 separate STANDARD repair candidates prepared, not trained. Defaults intact, no expected quality claims yet.

## Research audit and revised direction

User requires retaining mT5 while improving natural Singlish, student-conversation generalization and sensible replies. Read experiments/reply_research_v1.md and reply_research_v1_audit.json before proposing more runs. Audit:187/190 curated LOCAL targets differ from original normalized; Can interjection openings27→48/190, output20/30 hybrid; prior human both-bad19/30. Main proposed ablations on repaired same data/frozen STANDARD: full-context rewriting vs anchor-only vs anchor+training-only style exemplar. Rebuild pairs from appropriate natural LOCAL anchors with faithful STANDARD normalization; separate actual STANDARD-content repair. No new training launched; no reflexive RL or multiple-choice experiments. Original datasets/checkpoints intact.

## Human paired review negative

All30 fresh whole-pair reviews: hybrid5 wins, earlier paired5, tie1, BOTH BAD19. User appalled by unnatural Can prefix/STD copies; do not represent hybrid as an upgrade. Can prefix hybrid20/30 vs earlier14/30; exact casefold copies1 vs4 (near-copies excluded). Only1 per-case issue flag and no corrections, so unchecked issues are NOT passes. See experiments/hybrid_pair_ab_v1_results.json. No training/promotion.

## Fresh whole-pair A/B ready for user

Job355242 completed30 fresh cases, hybrid epoch1→rewrite versus earlier sequential paired. UI scripts/hybrid_pair_ab_test.py localhost8513; user votes pending in data/human_review/hybrid_pair_ab_v1/ratings.json. Four tests and actual UI smoke passed. Optional component issue flags/corrected pairs, no implicit pass labels or automatic training. See experiments/hybrid_pair_ab_v1.md. No default changes.

## Two-checkpoint STANDARD→LOCAL inference implemented

Optional V5_LOCAL_REWRITE_CHECKPOINT pairs epoch1 STANDARD model with earlier rewrite adapter. Job355188 completed53 pairs; STANDARD53/53 unchanged, anchors53/53 exact, actual app Candidate calls checked3/3. Twelve tests passed.14 assistant flags include clear refusal/actor changes; no quality-rate or superiority claim. See experiments/hybrid_inference_v1.md/results/review_flags. No training/default changes, checkpoints remote, Streamlit remains closed.

## Tiny structured diagnostic completed

Job354873 exit0. All modes fit12/12 training by120 updates. Final reorder/novel original3/36,3/12; plain1/36,2/12; seq2seq18/36,3/12. Only seq2seq learns to actually emit letters (others direct0 valid). Still poor generalization; clear tiny-set memorization. See experiments/structured_fit_v1_results.json. Next hypothesis randomized-position seq2seq warm-up with proper validation before RL. No new training/default changes.

## Tiny structured fit diagnostic submitted

Job354873 runs three supervised fit probes on12 cases: original regularized choice objective, plain choice CE, ordinary seq2seq letter CE.240 updates each, measure0/40/120/240;36 reordered same-fact cases and12 reused novel development cases, no test read. Both constrained-choice and directly generated letter metrics. See experiments/structured_fit_v1.md. Results pending, default unchanged.

## Structured RL pilot complete: no gain

Job354832 exit0. RL/SFT each ran96 updates, stopped after one epoch under patience1 as validation expected reward worsened. Both selected epoch0 baseline; identical synthetic test12/48 (25%),4/16 each category. No useful learning established. Inspect option-letter collapse and tiny training-fit capability before more RL. See experiments/structured_rl_v1_results.json. Default unchanged.

## Structured action RL pilot submitted

Job354832 compares REINFORCE categorical bandit, matched SFT control and unchanged epoch1 policy. Deterministic rewards on time/availability/ownership,192train/48val/48 synthetic test, max3epochs,p atience1; no free-form reply quality claim. See experiments/structured_rl_v1.md. Existing scorer is NOT used. Results pending, default unchanged, Streamlit stays closed.

## Reward revision2 audited: fresh gate failed

Jobs354827/354828 complete exit0. New scorer separate, original unchanged. Regression acceptance9/14→10/14, negatives23/23. Fresh48:accept5/11 good, reject23/24 bad, one positively rewarded cable ownership reversal; balanced70.64%, gate fails. No malformed calls. No RL actor updates; avoid training on this scorer. Full cases/results experiments/onpolicy_reward_v2.md/json. No scorer prompt/label/gate changes after results.

## Sampled reward audit failed

Completed jobs354821/354822.48 new sampled replies over12 scenarios; assistant labels frozen before scoring. Scorer rejects23/23 definite bad but accepts9/14 definite good, balanced82.1%, one schema inconsistency abstention. Frozen pilot gate failed despite9 mixed reward groups. Contextual false rejections include repeated refusal and tomorrow counterproposal; no RL submitted. See experiments/onpolicy_reward_v1.md and results JSON. Streamlit stays closed; default unchanged.

## Targeted scorer results retrieved

Job354811 completed exit0. Heldout24/24 allowed matches; actor23/25 (23 definite labels match,2 uncertain disagreements); calibration23/24; development23/24 with incorrect headphones permission rejection. Necessary diagnostic passed but not broad RL readiness; sampled on-policy audit next. No actor RL updates. See experiments/targeted_reward_v1_results.json.

## Targeted reward validation submitted

Job354811 freezes human-reviewed policy in prompted Gemma4 scorer;97 diagnostic calls. See experiments/targeted_reward_v1.md. User VPN verified up, SSH works. No RL actor updates; scorer results pending. Streamlit remains closed.

## Human DPO pilot submitted

Job354579 uses real A/B votes, epoch1 full checkpoint. 67 train/17 grouped validation preference pairs; max5 epochs, patience2, lr1e-5, beta.1, chosen-SFT anchor.1. See experiments/human_dpo_v1.md. Former validation34 conversations now TRAINING for this DPO model; do not report their outputs as held-out quality. Twenty fresh generation-only audit cases included. Default unchanged, results pending.

## Validation results available

All three standard_val_v1 jobs completed exit0. Full3e-4 best epoch1 val2.11055 versus original epoch8 val3.77084; baseline2.59882. Full1e-4 best epoch2 val2.12420; LoRA best epoch4 val2.15672. Clear late full-model overfitting. Reports local artifacts/standard_val_v1; see experiments/standard_val_v1_results.json. Default unchanged.

## Validation-controlled retraining submitted 2026-11-05

Jobs353450/353451/353449 audit prior endpoints and retrain LoRA3e-4/full1e-4/full3e-4 with train+val loss each epoch, best-val checkpoint including epoch0, patience2/max12. See `experiments/standard_val_v1.md`. Results pending, default unchanged.

## STANDARD scale jobs submitted (2026-11-04)

Jobs 352975 (LoRA3e-4), 352976 (full1e-4), 352974 (full3e-4), 790 STANDARD rows, 1584 updates. At submission verification: two running, full1e-4 queued by group active-job limit. Results not yet inspected; default unchanged. See `experiments/standard_scale_v1.md` and launch/recipe JSON. User requested ending after submission, no completion wait or monitoring.

# Role D — Transformer experiment (V5, mT5)

## STANDARD-only fit diagnostic complete — 4 November 2026

Four runs: q/v LoRA versus full fine-tuning, each at1e-4/3e-4,400 updates,24 checked STANDARD examples (10 user-reviewed +14 assistant-checked),24 wording-held-out paraphrases. Both full runs exactly fit24/24 by100 updates and reach24/24 training adequacy and18/24 paraphrase adequacy. LoRA3e-4 reaches21/24 and15/24; LoRA1e-4 reaches17/24 and11/24. Unchanged stage1 baseline is15/24 and11/24. This supports full STANDARD fine-tuning as a next experimental direction, not an intrinsic inability to learn. LoRA limitations are under tested settings; more steps/ranks/rates remain untested. Paraphrases retain scenarios/entities and are not independent new-scenario reasoning.

Report: `experiments/fit_probe_v1.md`; verified data/model/source hashes and counts: `experiments/fit_probe_v1_results.json`. Jobs352881/352882 complete; four final checkpoints retained. STANDARD-only diagnostic models do not replace paired-style default or establish LOCAL retention. One seed, subjective assistant adequacy plus exact-target annotations, no final-test reads, no DPO. Scaling full STANDARD training with a separate fixed LOCAL rewriter has not yet been run.

## All600 authorized additions trained — 4 November 2026

User instructed accepting everything and training:100 fresh plus500 existing candidates, with10 individually reviewed pairs and590 blanket-authorized additions (not600 human quality labels). Original190 plus600 yields790 conversations. Matched q/v sequential-rewrite control/expanded runs352816/352817 completed1120 updates on both GPUs. Expanded used all8 full epochs; effective source/target limits stayed192/48. Final checkpoint primary; best and step784 retained. No DPO preference pairs were fabricated.

Same50 validation cases: STANDARD36/50 both; Singlish38 control versus25 expanded; all-four28 versus16. Same24 diagnostics: STANDARD12 versus9; all-four3 versus7. Expanded has19 exact style copies/50 versus3 control. No general upgrade established; default unchanged, locked test untouched. These are one-seed assistant development ratings, not human accuracy. Full report: `experiments/blanket600_v1.md`; verified results: `experiments/blanket600_v1_results.json`. Expanded weights: personal account `~/ReplyCoachSG_blanket600_v1/artifacts/expanded_s425/final`.

## Training-duration probe completed — 4 November 2026

Control and targeted-data/broader-adapter trajectories each reached784 updates, with exact snapshots160/320/480/784. All eight variants were reviewed on24 newly frozen assistant-authored development cases. Earlier stopping did not improve STANDARD adequacy: final12/24 for both arms; best earlier11/24 control and12/24 broader. Broader all-four ties9/24 at480/784 and is6/24 at320 despite lower reference loss. Do not equate late reference-loss deterioration with a demonstrated earlier reply-quality upgrade. Default unchanged; no RL or held-out-test access.

Report: `experiments/duration_probe_v1.md`; hashes/results: `experiments/duration_probe_v1_results.json`. Group352662 finished training but failed text-reproduction validation; inference-only recovery352670 verified exact step784/final tensor identity and retained48/50 matching secondary pairs. Personal352663 completed normally. Preserve original logs, snapshots and all ratings; no bitwise real-GPU generation claim. One seed,24 development conversations, assistant labels, not human accuracy.

## STANDARD-quality experiments completed — 4 November 2026

Three approaches plus matched control and control/candidate seed1/2 replication completed: eight training runs, all784 updates. Tested80 targeted role/intent conversations, broader attention/feed-forward LoRA, and doubled STANDARD exposure while retaining sequential LOCAL rewriting. The combined data/broader-adapter candidate improves the24 authored diagnostic cases across seeds, but existing-validation STANDARD averages36.7/50 versus37.7/50 control; all-four averages28.7/50 versus28.0/50. Meaning preservation is worse on average. No general upgrade established; configured default unchanged, no RL or held-out-test access, both account queues empty after completion. All figures are exploratory assistant development ratings, not human accuracy.

Full report and limitations: `experiments/standard_focus_v1.md`; raw-report hashes and per-seed reviews: `experiments/standard_focus_v1_results.json`; combined-candidate replication: `experiments/standard_focus_v1_replication.json`. Group successful jobs352584/352587/352606; personal352599/352607. Preserve all best/final adapters and reviews. Do not treat selected diagnostic gains as a reason to promote or keep tuning against the same cases.

## Training ablations closed; evaluation handoff — 3 November 2026

Retain the existing supervised V5 checkpoint. The final incoming-register test
kept all 59 scenarios/target replies fixed and changed only input wording;
three new runs (job **350473**) completed successfully in **3m 26s** with
280 updates each. Mean validation loss: control **2.5068**, formal-input
augmentation **2.5308**, chat-register augmentation **2.5499**. Primary assistant
review: 19 control preferences vs 14 chat-register preferences, 35 ties,
7 both-bad and 25 identical. Reviews are non-independent and not fully blind.
No upgrade, selected-checkpoint change or final-test result is claimed.

Close further small training ablations rather than chase reused validation.
Consolidated decision and next evaluation deliverables:
`experiments/v5_ablation_summary_20261103.md`; frozen source/result hashes:
`experiments/v5_ablation_freeze_20261103.json`. New run report and review:
`experiments/synthetic_register_v1.md` and
`experiments/synthetic_register_validation_review.md`. All new adapters remain
archived in group-account experiment snapshots; no jobs remain running.

Final-evaluation rubrics now allow plausible new selectable-draft details,
reject explicit contradictions/role reversals and do not automatically prefer
agreeing over declining. **14 judging-runner tests passed**. Fresh one-persona
policy check **350486** matched **6/6** user-reviewed examples; this is a development
check, not general judge accuracy. Input exporter check passed: 50 validation
queries/190 bank items, no models loaded. Main multi-system/ten-persona validation
and end-to-end demo verification remain work, not completed scores. Keep old
ratings historical, use fresh matched runs with the updated rubric, and do not
read the locked test set until all systems/judging configuration are frozen.


## Paired synthetic-data pilot completed — 2 November 2026

The 60-candidate pilot produced **59** reviewed fictional conversations (118
style replies); one was held out for awkward LOCAL wording despite model
approval. Separate-model review job **349716** completed in 3m 10s; matched
training job **349763** completed in **5m 50s**, exit 0, group `CS425G9`, node
`lava`, one RTX 3090. All six models reached **280 optimizer updates**.

Mean validation loss: **control 2.5068 vs augmented 2.5308**, worse for augmentation
in all three seeds. Controls reproduced all previous control losses and replies
exactly. Primary seed-425 assistant review: **19 control preferences, 16 augmented,
25 ties, 10 both bad, 30 identical**. This is not fully blind or independent human
review: the assistant authored the data and had seen previous control replies.
Other seeds have quantitative comparisons, not additional qualitative ratings.

Decision: retain the existing supervised baseline. Preserve all six adapters,
source/review provenance and learning curves. No default checkpoint changed,
no jobs from this pilot remain running, and no held-out test content was read.
This is a negative result for **59 added conversations at a fixed update cap**,
not evidence that all synthetic data fail. The larger 190/570 volume arms were
not run. Mostly formal incoming synthetic dialogue is a possible domain mismatch;
a separate incoming-register ablation is a next hypothesis, not an established
cause. Do not expand volume or tune repeatedly on reused validation until a win.

Full report: `experiments/synthetic_paired60_v1.md`; primary per-case review:
`experiments/synthetic_paired60_validation_review.md`; recipe/hashes:
`experiments/synthetic_paired60_v1.json`. Local ignored results:
`artifacts/synthetic_paired60_v1/`. Cluster snapshot:
`~/ReplyCoachSG/experiments/synthetic_paired60_v1/`. The 42 targeted tests,
post-run budget/input checks and control reproducibility checks passed.


## Paired synthetic-data pilot running — 2 November 2026

User authorized continuing after the inconclusive DPO result. Authored **60**
fictional paired-style candidates across six scenario groups. Gemma paired review
job **349716** approved all 60; author audit held out one whose explanation
acknowledged awkward LOCAL wording. Reviewed training file contains **59**
conversations (118 STANDARD/LOCAL reply rows), not 59 preference labels.
No original starter candidate was silently promoted or rewritten.

Training job **349763** runs control versus synthetic augmentation, seeds 425/1/2,
on the group account from `~/ReplyCoachSG/experiments/synthetic_paired60_v1/`.
Both arms have the same **280 optimizer-update cap**, eight-epoch cap, patience
two, lr 3e-4, batch 16, stage-1 base and rank-16 LoRA. No DPO/unlikelihood.
Added data must not simply gain more updates; actual updates and partial-epoch
validation are recorded. Early stopping, exposure and checkpoint opportunity
counts can differ and must be disclosed.

Candidates/reviewed data: `data/synthetic/paired60_v1_{candidates,reviewed}.jsonl`.
Experiment/recipe: `experiments/synthetic_paired60_v1.{md,json}`. Lexical overlap
max 0.319 with no >=0.35 flag against train/validation; this is not a semantic
leakage guarantee. Reviewers are models, not independent human/native speakers.
42 targeted tests passed. No test-set content copied or read. Existing default
checkpoint and original training/validation files are unchanged.


## Matched DPO experiment completed — 2 November 2026

The user explicitly authorized continuing independently without further guidance.
Job **349499**, group account `CS425G9`, completed successfully in **8m 05s**
on `lava` with one RTX 3090 (24 GB VRAM). Six fresh LoRA runs were trained:
supervised-only and supervised+DPO, seeds 425/1/2, same base/data/learning rate,
eight-epoch cap and patience two. DPO used the reviewed 73-preference subset.

Mean best validation loss: **2.5068 control vs 2.4951 DPO**. Small lower loss in
all three seeds did not establish consistent reply-quality improvement.
Anonymized assistant review: **32 control preferences, 24 DPO preferences,
67 ties, 36 both-bad pairs and 141 identical outputs** across three uses of the
same 50 conversations × two styles. These are not independent human ratings,
300 independent conversations, a final test score, or overall bad-reply rate.
Seed 425 was primary; reviews of seeds 1/2 were exploratory follow-ups.

Decision: **retain the existing supervised baseline**. Preserve all six new
adapters as experimental artifacts; no default provider/checkpoint changed.
All runs reached the cap; five best epochs were eight, one seven. No sustained
validation-loss divergence was observed within this short run, which does not
rule out semantic overfitting or memorization. Peak GPU allocation: control
~5.7 GB, DPO ~9.9 GB. No jobs from this experiment remain running.

Full results: `experiments/dpo_review73_v1.md`; all anonymized-review decisions:
`experiments/dpo_review73_validation_review.md`; fixed recipe/hashes:
`experiments/dpo_review73_v1.json`. Local raw reports and learning curves:
ignored `artifacts/dpo_review73_v1/`. Cluster source/data/checkpoint snapshot:
`~/ReplyCoachSG/experiments/dpo_review73_v1/`. Source and original checkpoints
were not overwritten; no test-set content was copied or opened.

Next research priority: better paired task supervision in a separate data
ablation. Do not claim DPO improvement or tune repeatedly on reused validation
until it wins. Main persona evaluation and V4/V5/hybrid comparison remain
separate work; this diagnostic assistant review does not replace them.


## Retained-label review completed — 2 November 2026

All **94** critic-retained preferences were reviewed by the assistant with full
training context and both candidates. Proposed outcome: **73 keep**, **3 exclude**,
**18 borderline**. Borderline cases are held out, not declared objectively wrong.
The proposed 73 preferences are **37 STANDARD / 36 LOCAL**, covering **61**
training conversations. The review is not independent human gold: the user
reviewed six policy examples, not these 94 preferences.

Read `experiments/critic_full_v5_review.md` for all 94 per-case decisions,
contexts and rationales. Raw structured review is ignored
`artifacts/critic_full_v5_assistant_review.json`; proposed filtered training input
is `artifacts/critic_full_v5_reviewed_proposed.json`. Original critic output is
unchanged, and excluded/borderline target replies were not rewritten.

Notable issues: irrelevant workload answer for a badminton question; departure
misread as arrival in a zao-first exchange; weak STANDARD grammar; arbitrary
personal-status/compliance preferences and ambiguous style boundaries. Missing
new personal or factual details alone is not grounds for rejection.

No training was started by this review. Next prepare short matched supervised-only
versus supervised+DPO runs using the conservative proposed subset. Preserve the
locked test set and inspect validation outputs in both styles; independent
style preference training does not enforce equivalent meanings across styles.


## Full preference labeling completed — 2 November 2026

Job **349304** completed successfully (exit 0) in **1h 33m 31s** on `aloha`,
under `CS425G9`. All **380/380** comparisons processed; **94** preferences kept
(**53 STANDARD, 41 LOCAL**). Skipped: 155 order disagreements, 59 both-bad pairs,
64 ties, 6 uncertain screens and 2 uncertain pairwise judgments. Raw report
retrieved locally to ignored `artifacts/critic_full_v5_screened.json`.

This supersedes older running-job status below. Label generation is complete;
the full retained-label inspection is not yet complete. No DPO training was
submitted or automatically chained. Existing V5 weights are unchanged.
Next inspect retained/skipped decisions and label distribution before preparing
short matched supervised-only versus supervised+DPO runs. Do not call 94 kept
labels human-validated gold or evidence of improved reply quality.


## Update format preference

The user requests a clear, structured summary at the end of each work session
and status update: completed work, currently running jobs (account and job ID),
results with limits, next actions, and whether user input is needed. Distinguish
label generation from generator training; do not imply quality improved without
a completed model comparison.


## Authorized cluster accounts — 2 November 2026

The user explicitly authorizes using both the CS425 group account `CS425G9`
and their personal account `hm.teoh.2024` for this project's cluster experiments.
Both accounts connect to the same cluster host; they are not separate clusters.
Current job 349304 and existing V5 artifacts belong to the group account.
Use each account within its permitted allocations and scheduler limits; do not
use the second account to circumvent a group/project resource restriction.
Keep credentials in the existing sibling-project `.env`, never in these notes.


## Current experiment status — 2 November 2026

User-policy transfer check **349274** completed: isolated acceptance matched
24/24 candidate expectations across 12 fresh assistant-authored cases; this is
not independently reviewed accuracy. Screened training pilot **349291**
completed: 6 preferences retained, 7 both-bad pairs skipped, 5 order disagreements
skipped, 2 ties skipped. The assistant inspected all six retained preferences
and found no clear contradiction or role reversal; these labels are not human gold.

Full labeling job **349304** is running on `aloha` for all 380 training-only
comparisons, using the same v5 rubric and isolated selected-reply screen.
Output: group account `~/ReplyCoachSG/artifacts/critic_full_v5_screened.json`;
log: `slurm-v5-critic-349304.out`. Four-hour allocation limit, not a runtime estimate.
This is labeling only. No DPO job is submitted or automatically chained. No
ongoing monitoring is scheduled. Next inspect accounting, completion status,
counts and retained/skipped decisions before matched supervised-only versus
DPO training. Sync and verify the updated trainer before submitting training.

Reports: `experiments/critic_transfer_v5.md` and
`experiments/critic_pilot_v5_screened.md`. Current pairwise prompt is
`selectable-v5-user-reviewed`, isolated prompt `selectable-single-v3-user-reviewed`.
Older pause instructions and older stricter factual-label policies below are
historical and superseded by user resumption and subsequent explicit review.


## Additional user review — 2 November 2026 (current policy)

Job **349251** completed: the revised isolated critic agrees with all **6/6**
explicit user judgments. This is a development check after policy revision, not
a held-out accuracy estimate. See [the report](../../experiments/critic_user_review_v5.md).
Next use fresh diagnostics and inspect a small training-only pilot under this
policy; the earlier scores do not justify replacing the critic.

The user additionally marked an unprovided official deadline and an unprovided
professor-approval claim acceptable as selectable drafts. Six exact single-draft
judgments are now recorded in `data/training_critic/user_review_20261102.jsonl`.
Do not reject a plausible draft solely for adding room/deadline/third-party
information. Reject explicit contradictions and role reversals. Missing details
are distinct from contradicting stated details. The user selects/edits drafts.

Current pairwise rubric: `selectable-v5-user-reviewed`; isolated rubric:
`selectable-single-v3-user-reviewed`. This supersedes the earlier policy notes
below. Original diagnostic files/reports remain historical. New assistant-revised
copies are `calibration_selectable_v5.jsonl` and `calibration_modes_policy_v5.jsonl`;
only the six single drafts were actually reviewed by the user. Their provisional
pairwise rankings and all other cases must not be called user-validated gold.

Earlier critic-failure counts include assistant labeling mistakes and are not
accuracy estimates for the user's intended task. Recheck the current critic
before deciding to replace it. A six-case reviewed development check is useful
for policy alignment but cannot establish general critic reliability.


## User review correction — 2 November 2026

The user reviewed four single drafts: unknown assignment progress acceptable;
unknown tutorial room (Room B12) acceptable; contradicting an explicit
cannot-meet-before-eight constraint unacceptable; reversing the lunch-buying
roles unacceptable. Exact examples and review scope are recorded in
`data/training_critic/user_review_20261102.jsonl`.

The old diagnostics incorrectly treated the unprovided room as automatically
wrong. Their published counts measure agreement with an earlier assistant-authored
policy, not user-validated accuracy; they must not establish that the critic
fails the user's intended task. Preserve old cases, prompts and raw reports.
Do not silently relabel them or call all remaining labels human-reviewed.
Other boundaries, such as official deadlines and third-party actions, were not
individually reviewed. They need review rather than confident failure claims.

Current rubric is `selectable-v4-user-reviewed`; isolated rubric is
`selectable-single-v2-user-reviewed`. Unknown details alone do not make a
selectable draft wrong. Clear contradictions and reversed roles still do.
Old preference labels are incompatible with the new trainer version gate.
Reassess the policy and remaining expectations before choosing a different
critic or launching DPO. No new cluster job is launched by this correction.


Inference remains local. As of 2 November, the user confirms permission for both LLM-generated training examples and LLM critic preference labels; see [the current data and DPO experiment](../../docs/synthetic_v5.md). Train on `data/dev/train.jsonl` plus explicitly reviewed synthetic examples, validate on `data/dev/validation.jsonl`, never open `data/test/test.jsonl`.
Work branch: `role-d/v5-finetune`. Cluster commands need `--partition=project --account=cs425 --qos=cs425qos` (written `...` below).

## Resumed experiments — selectable drafts, 2 November 2026

Follow-up jobs **349195/349196** completed: isolated judging classified 42/48
retained and 20/24 fresh candidates correctly. It still incorrectly accepted
candidates in 5/24 retained and 3/12 fresh cases. This improves diagnostics but
fails the reliability gate; no automatic DPO labeling or training started.
See [the mode comparison](../../experiments/critic_modes_v1.md). The next useful comparison is a
different local critic under frozen conditions, rather than more tuning here.

Calibration job **349147** completed: 8/24 full passes, 21/24 order agreement,
15 cases with incorrect candidate acceptance; expected preference matched both
orders on 19/24 before flag/certainty checks. DPO expansion remains gated on
critic reliability. See [the run analysis](../../experiments/critic_selectable_v3.md).

The user resumed experiments and clarified that plausible new personal details
are useful in reply suggestions the user chooses before sending. The current
critic rubric is `selectable-v3`: allow unknown personal progress, availability,
preferences, companions, offers and proposals; reject explicit contradictions,
wrong actors, outdated agreements and unsupported precise shared-world answers.
The former blanket unsupported-personal-facts rule is superseded. Earlier
calibration scores describe that former task and must not be treated as current
critic accuracy. Old label files must be regenerated before DPO.

New diagnostics: `data/training_critic/calibration_selectable_v3.jsonl` (24
AI-authored cases, not independent human gold). Unknown progress now expects a
tie between plausible answers. Each case specifies candidate acceptability;
calibration checks these flags in both orders as well as preference accuracy.
A correctly ranked pair can still fail if the critic accepts its contradictory
alternative. Keep old diagnostic files and reports for provenance.

Next: inspect the new calibration, diagnose failures, then rerun and inspect the
20-comparison pilot before expansion to 380 or matched DPO training. Keep the
synthetic-data and architecture experiments separate. Test data remains locked.

## Earlier handoff — 2 November 2026

Further experiments are deferred at the user's request. Two critic jobs were
already submitted; they may finish normally. Do not submit more jobs, relabel
the full sheet or start DPO until work resumes.

- Existing supervised checkpoints remain the baselines. No new DPO model has
  been trained, and no quality improvement is established.
- Initial critic pilot: job **348615**, `artifacts/critic_pilot_v1.json`.
  During inspection it accepted unsupported companions/schedule claims in both
  reply orders. These labels are unsuitable for training; the trainer rejects
  their old prompt version.
- Revised critic calibration: job **348621**, last observed running on `aloha`.
  Output: `artifacts/critic_calibration_v2.json`; SLURM log:
  `slurm-v5-critic-348621.out`. It tests 12 AI-authored diagnostics, not independent
  human gold. Results have not yet been inspected.
- Ollama is installed under `~/.local/share/replycoach-ollama`; model cache is
  `~/.cache/replycoach-ollama/models`. Gemma 4 12B stays frozen; only the generator
  LoRA is updated by DPO. Both cluster accounts are on the same host; the group
  account holds these jobs and artifacts. Credentials stay in the sibling
  project's `.env`.
- Next session: retrieve both outputs and job accounting, inspect calibration
  failures, then rerun the 20-comparison pilot with the revised critic if sound.
  Expand to 380 only after checking label quality. Compare supervised-only
  against supervised + DPO with matched data, base and seeds.
- Later architecture experiments: shared versus separate/style-specific LoRA,
  then learned MoE-LoRA routing. Do not change architecture and preference data
  together. Neither style-specific adapters nor MoE-LoRA is implemented here.
- Checks completed: 33 targeted critic/fine-tuning tests passed after the latest
  changes. No test-set examples were opened.

## Historical pipeline snapshot

| Stage | Script | What it teaches | Status |
|---|---|---|---|
| 0 (optional) | `spans_v5.py`, `v5_spans.sbatch` | Read Singlish: span corruption on NUS SMS + CoSEM text | built, not run |
| 1 | `train_v5.py`, `v5_pretrain.sbatch` | Reply: DailyDialog (STANDARD) + CoSEM (LOCAL) pairs | done: small 3.466, base 3.345 |
| 2 | `finetune_v5.py`, `v5_finetune.sbatch` | Task and styles: LoRA on 190 dev items × 2 styles | done: base LoRA 2.511 (best), small LoRA 2.639, small full 2.621 |

Stage 2 problems seen on validation: LOCAL replies often not local (24/50 with a marker vs 35/50 in references), copying the context, mixing up who said what.

## Historical to-do list

The checklist below is historical: stage 0, augmentation, speaker labels,
unlikelihood, joint outputs and full-CoSEM runs already have local reports.
Use `MODEL_CARD.md`, `artifacts/*.json` and `docs/synthetic_v5.md` for the current
baseline and next experiment. Do not repeat this checklist as unfinished work.

### 1. Augmentation runs (each about a minute; compare with 2.511)
- [ ] `rsync -av --exclude .venv --exclude artifacts ./ CS425G9@10.193.104.102:~/ReplyCoachSG/`
- [ ] `NAME=base_lora_ctx BASE=artifacts/mt5_base/best sbatch ... scripts/v5_finetune.sbatch --context-variants`
- [ ] `NAME=base_lora_reh BASE=artifacts/mt5_base/best sbatch ... scripts/v5_finetune.sbatch --rehearsal-pairs data/public/dailydialog_train_pairs.jsonl data/public/cosem_train_pairs.jsonl`
- [ ] `NAME=base_lora_both BASE=artifacts/mt5_base/best sbatch ... scripts/v5_finetune.sbatch --context-variants --rehearsal-pairs data/public/dailydialog_train_pairs.jsonl data/public/cosem_train_pairs.jsonl`

### 2. Stage 0 chain (compare stage 1 with 3.345 and stage 2 with 2.511)
- [ ] Check: `NAME=spans_check sbatch ... scripts/v5_spans.sbatch --train-limit 2000 --epochs 1 --eval-every 50`
- [ ] `NAME=mt5_base_spans sbatch ... scripts/v5_spans.sbatch` (roughly 30–60 min)
- [ ] `NAME=mt5_base_after_spans MODEL=artifacts/mt5_base_spans/best EPOCHS=5 sbatch ... scripts/v5_pretrain.sbatch` (~2 h; EPOCHS=5 matches the original base run)
- [ ] `NAME=base_spans_lora BASE=artifacts/mt5_base_after_spans/best sbatch ... scripts/v5_finetune.sbatch` plus the best flags from step 1

### 3. Read the results
- [ ] On the laptop: `scp 'CS425G9@10.193.104.102:~/ReplyCoachSG/artifacts/*.json' artifacts/`
- [ ] For each run: epoch 0 vs best loss, STANDARD vs LOCAL loss, when it over-fit, LOCAL marker rate, how often it copies the input
- [ ] Pick the final V5 checkpoint

### 4. Connect and compare
- [ ] Fill in `load_v5()` in `src/replycoach/generate/providers.py`: base + adapter + tokenizer, `mt5_input` format, Candidates for both styles with `source="V5"` and a score in [0, 1] (e.g. exp of mean token log-prob)
- [ ] Decide where V5 runs for the demo (the laptop cannot load mT5-base comfortably)
- [ ] Export dev outputs: `python scripts/export_dev_outputs.py --generator v5 --output <file>`
- [ ] Compare with V4 on the same 50 validation items once Role C connects `load_v4()`; if V5 is not clearly better, V4 stays

### 5. Write-up
- [ ] Stage 0/1/2 design and why (self-supervised next-turn prediction vs span corruption, one tagged adapter vs two, LoRA vs full)
- [ ] Results tables, augmentation and stage 0 ablations, compute cost (GPU, minutes, peak memory)
- [ ] Limitations: 190 items, validation used for epoch choice and the V4 comparison, dated NUS SMS language, facts the model cannot know
- [ ] Ask Role A to add CoSEM, NUS SMS and mT5-base to `docs/licence_checklist.md`; cite Gonzales et al. (2023) and Chen & Kan (2013)
- [ ] Clean reproduction notebook (weeks 10–11)

### Set aside
- Llama-SEA-LION + LoRA: possible extra experiment later (8B, needs the 48 GB GPUs; the demo cannot run it on a laptop).

Commands and data steps: `docs/kaggle_notes.md`.
