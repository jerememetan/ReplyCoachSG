# Incoming chat-register ablation — 3 November 2026

Bounded follow-up to the 59-conversation synthetic pilot. Hypothesis: mostly
formal incoming dialogue may transfer poorly to shorthand/Singlish/code-switched
chat. This run tests input register while holding scenarios and reply targets
fixed. It does not add new conversations, change architecture or add DPO.

## Fixed before new output inspection

- Same 59 source IDs, speaker sequence, fictional situations, facts and requests.
- STANDARD/LOCAL reply targets are **byte-for-byte unchanged**.
- Only history/incoming wording changes to restrained chat syntax, abbreviations
  and some English–Mandarin code-switching. These are assistant-authored variants,
  not samples proving representativeness of real Singapore chats.
- Formal source: `data/synthetic/paired60_v1_reviewed.jsonl`; new candidates:
  `data/synthetic/paired60_register_v1_candidates.jsonl` (unapproved until reviewed).
- Local Gemma reviews original and rewritten input together plus both targets,
  checking source-meaning preservation and target compatibility. Unknown factual
  details remain allowed as selectable replies under user policy. This is model
  review plus author audit, not independent human/native-speaker validation.
- Quantitative comparison across seeds 425/1/2; primary qualitative seed **425**.
  Qualitative arm labels are hidden, but the assistant has seen past control/formal
  outputs, so do not claim fully blind or independent judgments.
- Same stage-1 base, rank-16 LoRA, lr 3e-4, batch 16, context pairs, 8-epoch cap,
  patience 2, **280-update cap**, no DPO/unlikelihood. Formal and chat augmentation
  both have 678 training rows and the same potential checkpoint schedule.
- Reuse completed control/formal runs only after verifying source/data hashes and
  shared training settings; no need to spend GPU time repeating unchanged runs.
  The previous controls already reproduced the preceding experiment exactly.
- This is the last planned small training ablation before freezing the baseline
  comparison and preparing final evaluation/report artifacts; do not keep tuning
  this reused validation set until a win appears.

## Review and overlap status

Register-review job **350463** submitted. It compares all 59 old/new situations.
Reviewer/version: `paired-register-review-v1`. The original 60-to-59 exclusion is
preserved. No scenario is silently removed or target edited to obtain approval.

All 45 source/train/validation/lexicon hashes used by the prior experiment were
verified unchanged, as was the original reviewed-file hash. Lexical TF-IDF maximum
overlap against train/validation is **0.195**, zero flags >=0.35. This method is
weak for code-switching and does not establish absence of semantic leakage.
Exact-normalized overlap checks will also run through the training loader.

14 paired-review/loader tests passed, including old/new input visibility and a
hard rejection if IDs, reply targets or speaker order change. No test file is read.

## Results

Review completed: **58 model approvals and one rejection**. The author audit
corrected the rejection for `roles-05`, whose unchanged reply promises to put
“your guitar” back after borrowing it. The reviewer incorrectly calls that user
ownership; both old/new inputs explicitly make it the peer’s guitar. This is a
documented author override, not a hidden model approval. Source candidates and
raw model rejection remain unchanged. No target was rewritten or scenario dropped.

The reviewed file contains all 59 original IDs/speaker sequences and exactly
unchanged reply targets. Model/author review is not human gold. Reviewed source:
`data/synthetic/paired60_register_v1_reviewed.jsonl`, SHA256
`65d46c99d9995f057adf7d3d7303e2a6cd98f6f24be9fe78826cd605b28cebfe`.
Raw reasons and the correction are retained in ignored
`artifacts/paired60_register_v1_{review,author_audit}.json`.

Three new runs are submitted from a frozen source/data snapshot; comparator
reports are reused with recorded hashes. New recipe/hashes:
`experiments/synthetic_register_v1.json`. No provider or selected checkpoint
is changed during training. Final model results pending.


## Completed results and decision

Input review **350463** completed in **3m 45s**. Three-run training batch **350473**
completed successfully in **3m 26s** on `toga`, group `CS425G9`, one RTX 3090
(24 GB VRAM). All runs reached exactly **280 updates**, ended in a partial seventh
epoch, and produced both styles for all 50 validation conversations. The same
59 scenarios/target replies were used in formal and register augmentation;
selected ID order and shared settings were verified by the analysis helper.

| Seed | Control | Formal augmentation | Chat-register augmentation |
| --- | --- | --- | --- |
| 425 | 2.4990 | 2.5332 | 2.5514 |
| 1 | 2.5075 | 2.5283 | 2.5463 |
| 2 | 2.5139 | 2.5309 | 2.5520 |
| Mean | 2.5068 | 2.5308 | 2.5499 |

Lower token loss is better. Chat-register input was worse than formal input in
all seeds: mean delta **+0.0191** (STANDARD +0.0043, LOCAL +0.0339). It was worse
than control by **+0.0431**. These are reused-validation token losses, not accuracy
scores or proof about all synthetic data or all Singlish inputs.

Primary assistant reply review: **19 control preferences, 14 register preferences,
35 ties, 7 both-bad, 25 exact identical**, across 100 style-specific cells. Full
context, replies and recorded reasons are in
[the review](synthetic_register_validation_review.md). These are subjective,
non-independent development judgments; no statistical superiority claim is made.
Other seeds have loss/diagnostic checks but no additional qualitative ratings.

Some role/direct-answer cases improve, including giving updates to an absent
classmate and handing over the user's charger. Some regress: a seven-o’clock
meeting confirmation becomes checking with the peer at seven, and a meal-location
question becomes a hunger statement. The input-style hypothesis does not yield a
reliable upgrade under this tested recipe/budget.

**Retain the existing supervised baseline and close exploratory training.**
Do not scale data volume or keep modifying the reused validation experiment to
force a win. This particular author-created style bundle is not a representative
sample of real Singapore chat. Twenty-two of the 59 incoming messages contain
Chinese characters; changes bundle abbreviations, syntax and code-switching.
Tokenization, length and attention patterns therefore change along with register;
there is no isolated causal claim about any one of those components.

All three best register checkpoints are at step 280/epoch 7. Their loss is still
improving at the cap; no longer-run overfitting conclusion follows. Fixed updates
do not equal fixed token/FLOP counts. Register and formal augmentation have the
same rows and checkpoint opportunities; control has fewer rows and different
per-example exposure. Peak allocated GPU memory is **5.49–5.55 GB**; recorded
epoch training/validation time is **36.4–36.5s per run**, excluding loading and
generation. No blank replies occur. Copy metrics remain heuristics, not semantic
error counts.

Artifacts: recipe/hashes `experiments/synthetic_register_v1.json`; reviewed data
`data/synthetic/paired60_register_v1_reviewed.jsonl`; ignored run reports, curves,
packet/key and ratings under `artifacts/synthetic_register_v1/`. All three new
adapters are retained on the group account at
`~/ReplyCoachSG/experiments/synthetic_register_v1/artifacts/register_s{425,1,2}/best`.
No selected checkpoint/provider/deployment changed and no training jobs remain.
The locked test set was neither copied nor opened.
