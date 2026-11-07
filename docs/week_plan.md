> Shared-intention replication **351130 complete** (12m55s). Meaning passes original→rewrite: seed425 17→42, seed1 21→48, seed2 23→48 (each out of50 reused development cases). All-three passes 8→23, 8→28, 9→23. Context remains about33/50 for both approaches. These are assistant judgments, not human accuracy. Next work should address STANDARD contextual errors; actual repeated failures are saved for review. Default unchanged.

> Shared-intention pilot **351086 complete** (6m43s). One-seed assistant paired review: original/aligned/rewrite same-meaning passes 17/19/42 out of50; all-three passes 8/10/23. Context passes 35/32/32, so context is still unresolved. These are fallible assistant judgments, not human accuracy. Replication job **351130** submitted for seeds1/2 with identical neural source and data. Default unchanged.

> Shared-intention implementation is ready: mT5 generates STANDARD, then rewrites that actual draft into LOCAL using the same model. Stronger paired targets are assistant-audited for all 190 original inputs. Three-arm, one-seed fixed-step pilot **351086 is RUNNING**; final outputs will be reviewed as pairs against the user's three criteria. See [shared-intention pilot](../experiments/shared_intent_v1.md). Default model unchanged.

> Primary quality requirements clarified by user: LOCAL must be recognizably natural Singlish; STANDARD/LOCAL should have approximately the same meaning; both must make sense in context, without needing to match the reference. See [requirements](model_quality_requirements.md). Prior rankings have not been re-scored under the stronger Singlish criterion.

> User objective clarified: **improve and deploy mT5; do not replace it with Gemma 12B**. Gemma is an offline teacher/comparison target only. Distillation job **351013** completed in 17m59s on one RTX 3090: original targets versus teacher targets versus a mix, same 190 inputs, three seeds, 784 updates each. All 190 teacher pairs were assistant-audited, 29 edited; preflight token/source/role checks passed. Mean losses control/teacher/mixed: 2.4959/2.9072/2.5375. Single-seed assistant review shows no established improvement; teacher checkpoint-selection bias is documented. See [mT5 distillation](../experiments/mt5_distill_v1.md).

> Latest experiment update: instruction-chat baseline and RL-readiness checks (2026-11-03). The 12-run +500 experiment is complete; no demonstrated synthetic-volume benefit. A Gemma chat baseline was preferred in 54 of 100 one-seed assistant comparisons against longer-trained mT5 (two mT5 wins, 42 ties, two both bad), with material model-size/pretraining/decoding confounds. Constructed reward diagnostics agree with 74/80 assistant labels but fail the frozen role-sensitivity gate. RL remains deferred; no default checkpoint was replaced. Details and limitations: [RL readiness](../experiments/rl_readiness_v1.md), [volume results](../experiments/synthetic_volume500_v1.md). Actual-output scorer diagnostic is complete: 56/80 agreement with fallible assistant expectations; the gate failed, and disagreement review found errors in both scorer judgments and reference labels.

# Remaining plan — transformer addendum 2 November 2026

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
a held-out accuracy estimate. See [the report](../experiments/critic_user_review_v5.md).
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


## Resumed experiments — selectable drafts, 2 November 2026

Follow-up jobs **349195/349196** completed: isolated judging classified 42/48
retained and 20/24 fresh candidates correctly. It still incorrectly accepted
candidates in 5/24 retained and 3/12 fresh cases. This improves diagnostics but
fails the reliability gate; no automatic DPO labeling or training started.
See [the mode comparison](../experiments/critic_modes_v1.md). The next useful comparison is a
different local critic under frozen conditions, rather than more tuning here.

Calibration job **349147** completed: 8/24 full passes, 21/24 order agreement,
15 cases with incorrect candidate acceptance; expected preference matched both
orders on 19/24 before flag/certainty checks. DPO expansion remains gated on
critic reliability. See [the run analysis](../experiments/critic_selectable_v3.md).

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

## Transformer handoff

The user permits LLM-generated training data and LLM critic labels. Current
details, commands and limitations are in [the V5 experiment plan](synthetic_v5.md)
and [Role D's handoff](../roles/D_transformer/START_HERE.md).

The first critic pilot accepted unsupported personal facts despite agreeing in
both display orders. Its labels are blocked from DPO. A revised critic is being
tested on 12 diagnostic cases (job 348621); results have not been inspected.
The critic is frozen, and no DPO generator training has started. Existing
supervised checkpoints remain the baselines.

At the user's request, leave further experiments for later. Already-submitted
jobs may finish; no additional jobs are needed now. On resumption: inspect
calibration, rerun the preference pilot, expand labeling if reliable, and compare
supervised-only against supervised + DPO. Synthetic-data volume and style-specific
or MoE-LoRA adapters remain separate later ablations. Keep evaluation ratings
separate from training labels and preserve the locked test set.

## Previous plan — 17 October 2026

The original schedule below is historical. The current priorities and evaluation
method supersede its outdated data targets and manual-rating workload.

The 300-item reviewed split is published: 190 train/reply-bank items, 50 shared
validation queries and 60 untouched test items. Generator providers and real hybrid
quality results remain pending. E's local judging runner exists; main ratings and
independent pilot quality checks remain work, not completed results.

## Current evaluation decision

Following professor approval reported by Jereme, replace human output reviewers
with **ten initial configurable persona-based AI reviewers**. Each keeps the same
persona and rates every compared reply on **appropriateness, naturalness and
style fit**, each 1–5, with an explanation and problem/uncertainty flags. This is
not a three-specialist panel or a human survey. It also records would-send yes/no
and reports its simulated percentage. Jereme/E now owns the full judging workflow;
A retains frozen-data stewardship and supports independent pilot sanity checks.
Dataset authorship and independent
human peer review remain unchanged. See `split_and_rating_protocol.md` for the
current protocol, local command, limitations and remaining evaluation work.

## Remaining milestones

| Dates | Deliverable |
| --- | --- |
| Now–23 Oct | C/D real-model handoffs; B retrieval/memory baselines; E local judge setup and five-conversation/ten-persona validation pilot, with A supporting independent checks; E export handoff and real hybrid verification when a model arrives. Choose the generator using validation quality and compute. |
| 26–30 Oct | Main validation evaluation; analyse persona disagreements and system errors; make evidence-based targeted improvements. Draft the report in parallel. |
| 2–6 Nov | Freeze systems, judge/persona configuration and rubric; evaluate the 60 held-out items once; finish tables, 15–20 qualitative cases, slides and the 20-conversation demo. |
| 7–9 Nov | Rehearsal and blocking bug fixes only. |
| 10 Nov | Final presentation with at least 90% of results complete. |
| By 27 Nov | Final report, code/data/reproduction package and peer evaluations. |

No evaluation results are established by this plan. E must check privacy, the
local judge's reliability and runtime before running the full panel. Do not tune on test data
or use AI ratings as training labels.

## Historical schedule

Snapshot as of 29 Sep 2026. Week 4 Friday was 18 Sep. Status cells and outstanding
data work below describe that earlier point, not current completion.

| Week | Dates (2026) | Friday proof | Status in this repo |
| --- | --- | --- | --- |
| 4 Foundation | 14–18 Sep | V0 retrieves a reply; shared format | Done as a head start |
| 5 Data + first models | 21–25 Sep | Authoring and peer review underway; V1 and V3 outputs | V1 works; V3 one-batch works; team must author data |
| 6 Core upgrades | 28 Sep–2 Oct | 300 items authored; V2, V4, basic V6 | V6 baseline exists; V2/V4 stubs exist; proposal due 2 Oct |
| 7 Generator decision | 5–9 Oct | Choose V4 or V5 | Comparison script still needed |
| 8 Final hybrid | 12–16 Oct | E2E demo on 20 unseen chats | Pipeline already wired; polish later |
| 9 Evaluation | 19–23 Oct | Tables and graphs | Forms to fill |
| 10 Targeted improvement | 26–30 Oct | One retrieval fix, one generator fix | Not started |
| 11 Freeze | 2–6 Nov | Submission-ready | Presentation 10 Nov |
| 12 Buffer | 9–13 Nov | Bugs only | Report 27 Nov |

Historical authoring target (now completed; do not use the 20 fixtures as gold):

- Each person: 60 unassigned items and one independent peer-review set of 60 items
- Review cycle: A reviews B, B reviews C, C reviews D, D reviews E, and E reviews A
- Team total: all 300 items marked `completed` after a valid review
- After all 300 are completed, Role A assigns the fixed, balanced 80:20 split of 240 development and 60 untouched test items
