# Isolated versus pairwise critic — 2 November 2026

## Additional user review — 2 November 2026 (current policy)

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


Question: does assessing one candidate at a time improve acceptability judgments
compared with the combined pairwise preference/acceptability request?

## Design fixed before results

- Retained diagnostics: 24 cases, job **349195**, `critic_single_retained_v1`.
  Reuse job 349147 pairwise results only if the case hash, rubric, schema, model
  digest and runtime settings match. Run two isolated requests per case.
- Fresh diagnostics: 12 cases frozen in
  `data/training_critic/calibration_modes_fresh_v1.jsonl`, job **349196**,
  `critic_modes_fresh_v1`. Run pairwise in both orders and two isolated requests.
- Same Gemma model and runtime options as v3. No validation/test items enter
  these diagnostics. All cases are AI-authored, not independent human gold.
- Preserve selectable-draft policy: plausible unknown personal details are allowed;
  explicit contradictions, wrong roles and unsupported precise shared-world
  answers are rejected. No prompt tuning based on these results during this run.
- Isolated output asks for a reason followed by an enum verdict, rather than
  ranking followed by two boolean flags. This changes both task decomposition
  and response schema/order; improvement cannot be attributed to isolation alone.
- Compare acceptance accuracy separately from preference accuracy. Report original
  and swapped pairwise results separately for equal candidate counts: the union
  of pairwise errors across two orders has twice as many opportunities to fail
  as a single isolated pass. Preserve the existing union metric for continuity.
- Isolated judgments do not rank two acceptable replies. This diagnostic does
  not demonstrate a complete DPO labeler, even if acceptance improves.

## Decision rule

Inspect incorrect acceptance and rejection as well as uncertainty. Do not use
these diagnostic outputs as preference labels. If rejection remains unreliable
on these simple cases, do not proceed to automatic DPO labeling. If promising,
check ambiguous cases independently and inspect a training-only pilot combining
acceptability screening with pairwise ranking before considering expansion.

## Results

Both runs completed and their raw reports were retrieved into ignored
`artifacts/critic_single_retained_v1.json` and `artifacts/critic_modes_fresh_v1.json`.
No DPO model was trained. Table entries count correctly classified candidates,
independent of preference ranking; all columns have equal candidate counts.

| Diagnostic set | Candidates | Pairwise original order | Pairwise swapped order | Isolated verdict |
| --- | --- | --- | --- | --- |
| Retained | 48 | 34/48 | 39/48 | 42/48 |
| Fresh | 24 | 22/24 | 18/24 | 20/24 |

Validation: 18 focused tests passed for mode-comparison isolation, malformed
responses, uncertainty, baseline provenance, existing critic behavior and trainer
compatibility. Shell syntax and Python compilation passed.


## Findings and decision

- Retained set: isolated judging got both candidates correct in **19/24** cases;
  incorrect acceptance occurred in **5/24** cases (six candidates).
- Fresh set: isolated judging got both candidates correct in **9/12** cases;
  incorrect acceptance occurred in **3/12** cases (four candidates).
- Neither isolated run produced uncertainty or incorrectly rejected an acceptable
  candidate. This is strong leniency on these cases, not calibrated confidence.
- Fresh errors reproduce wrong speaker roles, unsupported exam venues and
  unverified third-party actions. Retained errors also include a direct
  contradiction about companions. Raw explanations sometimes assert accuracy
  despite absent evidence or invent an interpretation to excuse a contradiction.
- Combined isolated candidate accuracy: **62/72**. These cases are small,
  AI-authored and not a population accuracy estimate. Fresh cases were frozen
  before this run's outputs, but authored by the same assistant after seeing
  earlier failure categories; they are not independently collected gold.

The simpler task/schema improves acceptance classification on retained cases.
On fresh cases it falls between the two pairwise orders (20/24 versus 22/24
and 18/24), so it does not establish a consistent generalisation benefit. It
does not satisfy the existing no-incorrect-acceptance gate.
Do not launch automatic DPO labeling from this critic. Isolated judging alone
also supplies no ranking between two acceptable replies. The current experiment
is complete; no automatic follow-up or hidden training job is scheduled.

Next useful experiment: compare a different locally runnable critic under the
same frozen policy and cases, with an independently reviewed diagnostic subset
before trusting any winner. Preserve these raw results; avoid repeated tuning on
these diagnostics. Reviewed synthetic supervision remains a separate route.
