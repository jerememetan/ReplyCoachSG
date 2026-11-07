# Selectable-draft critic calibration — 2 November 2026

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


Job: **349147**, group account, RTX 3090 node `aloha`. Rubric: `selectable-v3`.

The user resumed experiments and permits plausible personal details in selectable replies. Unknown progress, availability and preferences are allowed; explicit contradictions, wrong roles and unsupported precise shared-world answers remain failures.

## Result

- Completed 24 AI-authored diagnostic cases in both candidate orders. These are not independent human gold.
- Expected preference matched in both orders: **19/24** (before requiring acceptance flags and certainty).
- Candidate-order agreement: **21/24**.
- Full pass, including per-candidate acceptability and required certainty: **8/24**.
- At least one rejected candidate incorrectly accepted: **15/24**. The existing JSON field is named `unsafe_acceptance`; it includes non-answers/style failures as well as contradictions.
- Unknown assignment progress and unknown availability both passed as acceptable alternatives.
- This stricter rubric and scoring differ from v2; do not compare 8/24 with the earlier 11/12 as a model-regression estimate.

## Interpretation and next experiment

The critic often picks the correct candidate but labels both acceptable, sometimes despite a rationale describing a contradiction. Preference accuracy alone hides this failure. Do not use this run as preference labels or launch DPO from it. No quality improvement to V5 is established.

Next isolate individual-candidate acceptability judgments from pair ranking, test consistency between the two, and evaluate on fresh diagnostics in addition to retained cases. Do not simply repeat prompt tuning until this small set passes. Independently check ambiguous Singlish expectations. Only after reliable calibration inspect a new 20-pair training pilot, then consider expansion to 380 and matched supervised-only versus DPO training.

Existing supervised checkpoints remain unchanged. No validation/test conversations were used for these diagnostics. The test set remains locked.

## Per-case result

| Case | Full pass | Incorrect acceptance | Acceptance errors across both orders |
| --- | --- | --- | --- |
| CAL-001: unsupported_companions | False | True | 2 |
| CAL-002: unsupported_schedule | False | False | 1 |
| CAL-003: known_schedule | False | True | 1 |
| CAL-004: known_location | True | False | 0 |
| CAL-005: role_reversal | False | True | 2 |
| CAL-006: unknown_progress | True | False | 0 |
| CAL-007: prospective_offer | True | False | 0 |
| CAL-008: known_negation | False | True | 1 |
| CAL-009: local_naturalness | True | False | 0 |
| CAL-010: standard_style | False | True | 1 |
| CAL-011: changed_plan | False | True | 1 |
| CAL-012: missing_room | False | True | 2 |
| CAL-013: contradicted_progress | False | True | 1 |
| CAL-014: new_plan | False | True | 2 |
| CAL-015: unknown_availability | True | False | 0 |
| CAL-016: known_constraint | True | False | 0 |
| CAL-017: shared_deadline | False | True | 2 |
| CAL-018: known_room | False | True | 1 |
| CAL-019: new_preference | True | False | 0 |
| CAL-020: negated_offer | False | True | 1 |
| CAL-021: no_required_particle | True | False | 0 |
| CAL-022: third_party_claim | False | True | 2 |
| CAL-023: known_companion | False | True | 2 |
| CAL-024: useful_offer | False | True | 1 |

Raw report: ignored `artifacts/critic_calibration_selectable_v3.json` (retrieved from the cluster).

Validation: 49 targeted tests passed across critic/synthetic loader/fine-tuning/comparison; the final calibration-input validation edit was followed by another passing critic test run (11 tests).
