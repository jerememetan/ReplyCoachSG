# V5 data experiment — 2 November 2026

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


The user confirms permission to use both LLM-generated training data and LLM
critic preference labels. Earlier
statements prohibiting it describe the previous experiments. Existing checkpoints
remain non-synthetic; update their model cards only for newly trained versions.
Do not use evaluation persona ratings as training labels.

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

## Historical handoff: further work deferred

The user requested documentation updates and will revisit the experiments later.
Do not launch further experiments now. Already-submitted jobs may finish normally.

| Job | Purpose | Artifact on group account | Last known status |
| --- | --- | --- | --- |
| 348615 | Initial 20-comparison critic pilot | `artifacts/critic_pilot_v1.json` | No longer listed in queue at last check; final artifact/accounting not retrieved. Inspected partial decisions include unsafe acceptance. Blocked from DPO. |
| 348621 | Revised critic, 12 diagnostic cases, both orders | `artifacts/critic_calibration_v2.json` | Running on `aloha` at last check; results not inspected. |

Logs: `slurm-v5-critic-<jobid>.out` and `artifacts/ollama-<jobid>.log`.
Gemma 4 12B remains frozen. No DPO generator training or MoE-LoRA experiment has
been launched. The revised prompt and acceptance checks reduce known risks but
do not establish critic accuracy. Latest targeted checks: 33 tests passed.

When resuming, inspect queue/accounting and retrieve both JSON outputs before
claiming completion or using any labels. Check calibration's unsafe acceptance
and individual rationales; diagnose failures before relabeling the 20-item pilot.
If satisfactory, expand to 380 comparisons and run a matched supervised-only
versus supervised + DPO comparison. Separate synthetic-volume experiments from
architecture experiments. Try shared versus style-specific adapters before a
learned MoE-LoRA router; neither additional architecture is implemented yet.

The main bottleneck is paired task supervision: 190 authored conversations,
versus extensive unpaired public dialogue. The model card records role confusion,
generic answers and noisy CoSEM adjacency pairs. More raw CoSEM is unlikely to
resolve these alone. The saved full-CoSEM run has loss 2.4093 versus 2.4940 for
spk1_ul025_s425, but their recorded blind comparison is inconclusive. The joint
run's loss uses a different target format and is not directly comparable.

## First experiment

### Existing preference-training route (DPO)

DPO is already implemented in `finetune_v5.py` via `--preferences`,
`--dpo-weight` and `--dpo-beta`. It combines ordinary supervised loss with a
preference loss. `compare_v5 make-prefs` builds blind comparisons between beam
and sampled replies on training items. The local `artifacts/ab_prefs1.json`
contains 380 questions, all unanswered at inspection; no local run report has
nonempty preference-training metadata. This is implemented infrastructure,
not evidence of a completed DPO experiment.

A critic here supplies preferred/rejected reply pairs. The user explicitly
confirmed permission for both LLM generation and LLM critic labels on 2 November.
DPO itself needs no
separately trained critic or reward model. The existing echo unlikelihood runs
are a different objective: they penalize detected echoes rather than learning
from general reply preferences.

Before expanding synthetic volume, the existing 380-question sheet is a useful
small preference pilot: review training-only pairs for answering the message,
correct speaker roles, supported facts and style naturalness. Keep ties unused;
when both candidates are bad, the current loader uses the authored reply as
preferred, so check that reference fits the context too. The current terminal
workflow expects human choices. `scripts/critic_preferences.py` now provides a
separate training-only LLM workflow using the existing local Ollama transport.
It verifies each conversation against the frozen training file and records the
model digest, settings, rubric, source hashes and both raw judgments. Each pair
is judged in both orders; uncertain/disagreeing decisions, ties, identical
replies and both-bad pairs are excluded. Both-bad cases do not automatically
promote an unjudged authored reference. Incomplete/failed critic runs are
rejected by the trainer. Do not reuse held-out evaluation persona ratings.

```bash
python scripts/critic_preferences.py artifacts/ab_prefs1.json \
  --output artifacts/prefs_llm_pilot.json --check-only
# Requires a running local Ollama with the configured model installed:
python scripts/critic_preferences.py artifacts/ab_prefs1.json \
  --output artifacts/prefs_llm_pilot.json --limit 20
NAME=dpo_pilot SEEDS="425 1 2" BASE=artifacts/mt5_base_after_spans_spk/best \
  sbatch --partition=project --account=cs425 --qos=cs425qos scripts/v5_finetune.sbatch \
  --speaker-labels --context-pairs --preferences artifacts/prefs_llm_pilot.json \
  --dpo-weight 0.1 --dpo-beta 0.1
```

The DPO weight is a pilot setting, not a proven optimum. Inspect the 20 pilot
decisions and skipped cases before labeling all 380 (`--limit 380`, new output
path). Order agreement does not establish correctness or Singlish competence.
Compare the same supervised-only recipe with the preference flags omitted.
Local Ollama was unavailable during initial setup; subsequent cluster pilot
findings and the revised calibration workflow are recorded below.

### Cluster critic runtime

Ollama can run on the allocated GPU within a SLURM job. The wrapper
`scripts/v5_critic.sbatch` starts its own loopback server, pulls the configured
model into a persistent account cache, labels the pilot, records model residency
with `ollama ps`, and shuts down the server when the allocation ends. No inference
runs on the login node. `scripts/install_cluster_ollama.sh` performs a user-local
installation using the [official Linux archive](https://github.com/ollama/ollama/blob/main/docs/linux.mdx).
The [configured Gemma 4 12B model](https://ollama.com/library/gemma4:12b) is roughly
8 GB quantized; verify actual residency and runtime in the pilot log.

```bash
# On the group account, from ~/ReplyCoachSG (one-time runtime installation):
bash scripts/install_cluster_ollama.sh
NAME=critic_pilot_v2 sbatch --partition=project --account=cs425 --qos=cs425qos \
  scripts/v5_critic.sbatch
```

Copy the new critic script, its existing `judge_replies.py` transport, the SLURM
wrapper and `artifacts/ab_prefs1.json` to the cluster first. The script validates
the local frozen training content before making any model requests. For a full
run, set `CRITIC_LIMIT=380` and a new `NAME` after inspecting the pilot.

Compare a matched supervised-only control against supervised + DPO using the
same base, data and seeds. Tune DPO weight/beta on validation, inspect generated
replies rather than DPO agreement alone, and evaluate synthetic-data expansion
as a separate arm. The current trainer starts a new adapter on the stage-1
base; continuing an existing stage-2 adapter would require a loading change.

### Paired synthetic-data route

Keep mT5-base, speaker labels, separate style tags and the current training
recipe fixed. Compare no synthetic data, 190 reviewed synthetic conversations,
then 570, with seeds 425, 1 and 2. Use the same stage-1 base for every arm. Start
with the existing 190 authored examples plus 190 synthetic examples; increase
only if appropriateness improves without losing naturalness. Synthetic examples
are added once per epoch, not used to replace authored data. The current default
30 epochs and learning rate are experimental settings, not established optima
for a larger dataset; inspect early stopping and both style losses.

Generate 600 candidates in batches of 20, balanced across six groups: requests
and offers (who does what), direct factual questions, corrections and changed
plans, polite refusals, context-dependent references, and missing information.
Vary incoming English/Singlish independently from the requested output style.
Include double texts, zero to six memory turns, and restrained local language.
Target different situations, not paraphrases of the same template.

Review a 60-item pilot before expanding. Check that both replies make exactly
the same commitments, preserve negation and agency, answer the incoming message,
and invent no personal facts. A knowledgeable Singlish speaker should check local
naturalness. Do not require every LOCAL reply to end in lah/lor/leh. Automatic
marker rate is a diagnostic, not the quality objective.

`data/synthetic/pilot_candidates.jsonl` contains 12 initial LLM-written candidate
pairs to inspect and expand. They are deliberately unreviewed and the loader
will reject them until a separate reviewer records the checks. This is a starter
batch, not an evaluated dataset or evidence of model improvement.

## Generation prompt

Use this with an approved LLM; supply a scenario group and a unique batch ID.
Do not supply validation/test examples or real private messages.

```text
Write 20 diverse fictional Singapore university peer-chat training examples.
Generate a conversation and decide one useful, fact-supported next reply first.
Then express that SAME reply in standard conversational English and natural,
restrained Singlish. Preserve the actor, recipient, time, place, quantities,
negation, uncertainty and strength of commitment in both styles. Do not copy
the incoming speaker's first-person claims into the responder's mouth.
If a fact is missing, ask a useful clarification rather than inventing it.
Avoid caricature, particle stuffing and repetitive okay/no-worries openings.
Include varied speech acts and local syntax/vocabulary where appropriate.
Incoming message is always from peer; output is always the user's next reply.
Return JSONL only, using this exact structure and unique SYN- IDs:
{"id":"SYN-batch01-001","session_memory":[{"speaker":"peer","text":"..."},{"speaker":"user","text":"..."}],"message":"...","replies":{"standard":"...","local":"..."},"provenance":{"generator":"ACTUAL_MODEL_NAME","prompt_version":"paired-v1","batch_id":"batch01"},"review":{"reviewer":"","meaning_preserved":false,"speaker_correct":false,"natural_local":false,"facts_supported":false}}
Leave every review flag false: generation does not count as review.
```

After a separate review, record the reviewer and mark each check true only when
it passes. Keep rejected examples outside the training file. The loader checks
schema, provenance, paired styles, review attestations, duplicate IDs and
normalized conversation/message overlap with train/validation. It cannot verify
truthfulness of review flags, semantic equivalence or paraphrase leakage. Never
read the locked test set to filter or generate training data.

```bash
python -m replycoach.generate.synthetic_v5 data/synthetic/reviewed.jsonl
NAME=syn190 SEEDS="425 1 2" BASE=artifacts/mt5_base_after_spans_spk/best \
  sbatch --partition=project --account=cs425 --qos=cs425qos scripts/v5_finetune.sbatch \
  --speaker-labels --context-pairs \
  --negatives artifacts/candidates_train_spk1.json --unlikelihood-weight 0.25 \
  --synthetic-items data/synthetic/reviewed.jsonl --synthetic-count 190
```

Export PYTHON_MODULE as required by the existing cluster wrapper. Use new NAMEs
for each arm. Omit the two synthetic flags for the matched control. The report
records file SHA256, provenance file path and selected IDs; keep that exact file.
Synthetic examples also support --joint, but compare that in a separate ablation.

## Decision and remaining work

### Pilot finding: critic grounding needs calibration

The initial cluster pilot (`critic_pilot_v1`, job 348615) accepted unsupported
companions and schedule claims in both display orders. It also accepted local
vocabulary in a STANDARD reply. These observations establish that order agreement
alone is insufficient. Do not use this first pilot's labels for DPO; the trainer
rejects LLM critic metadata without the new `grounded-v2` prompt version.

The updated rubric requires an absolute acceptability check per candidate,
rejects unsupported personal facts, and distinguishes prospective offers from
claims about established facts. `data/training_critic/calibration_v1.jsonl`
contains 12 AI-authored diagnostics for grounding, roles, negation, changed plans,
style and valid offers. They are not independent human gold or final evaluation.
`scripts/calibrate_critic.py` keeps expected answers out of the critic prompt,
tests both orders and reports unsafe acceptance separately from accuracy.
Calibration outputs cannot be consumed as preference sheets.

```bash
NAME=critic_calibration_v2 CALIBRATION_INPUT=data/training_critic/calibration_v1.jsonl \
  sbatch --partition=project --account=cs425 --qos=cs425qos scripts/v5_critic.sbatch
```

Next: inspect calibration errors, require no unsafe acceptance on these simple
cases before another training pilot, and verify natural Singlish decisions with
independent checks. Passing these cases is only a diagnostic gate. Then relabel
the 20-item pilot with v2, inspect it, expand to 380, and compare supervised-only
against supervised + DPO using identical data and seeds. Keep MoE-LoRA as a later
architecture ablation so preference-data errors are not confounded with routing.

Evaluate on the fixed 50 validation conversations with blinded, shuffled output
comparisons. Record appropriateness, naturalness and style fit separately, plus
role reversals, unsupported facts, unanswered questions, echoes, and disagreement
in meaning between styles. Review disagreements and report uncertainty; do not
choose a model on LOCAL marker count or token loss alone. Keep the 60 test items
untouched until the final system and judging setup are frozen.

Only after this data ablation consider model changes. The joint output option
already exists and is a useful separate experiment for style agreement. Changing
architecture, data and decoding together would hide which change helped.
No new training or synthetic quality result is claimed by this plan.

Cluster inspection on 2 November: CLUSTER1 and CLUSTER2 in the sibling project's
.env are accounts on the same host, not two distinct GPU machines. Both queues
were empty; the group account holds the existing V5 checkpoints. Credentials
remain in that .env and are not copied here.
