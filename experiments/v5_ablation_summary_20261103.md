# V5 ablation conclusion and evaluation handoff — 3 November 2026

**Keep the existing supervised V5 baseline.** None of the three small training
ablations established a reliable reply-quality upgrade. Exploratory training for
these recipes is closed; all artifacts are retained, no jobs remain running,
and the locked test set is untouched. No provider or selected checkpoint changed.

| Training recipe, seeds 425/1/2 | Mean best validation token loss | Practical result |
| --- | --- | --- |
| Matched supervised control | 2.5068 | Reference comparator |
| Add 73 reviewed DPO preferences | 2.4951 | Slight lower loss, mixed reply quality; no upgrade |
| Add 59 formal-input paired synthetic conversations | 2.5308 | Worse loss in every seed; mixed replies |
| Same 59 scenarios/targets, chat-register inputs | 2.5499 | Worse loss in every seed; no reliable upgrade |

These are reused-validation token losses, not accuracy percentages. DPO training
has extra forward-pass cost; equal optimizer updates are not equal GPU compute.
The original selected `spk1_ul025_s425` checkpoint used a different historical
recipe, so its old loss is not inserted as another matched-control row.

Primary seed-425 assistant preferences: DPO 7 vs control 11; formal augmentation
16 vs control 19; chat-register augmentation 14 vs control 19. Different reply
sets have different identical/tie/both-bad counts. These counts must not be ranked
as directly comparable independent accuracy estimates. Reviews are assistant
judgments, not human gold; later reviews have prior exposure to comparator text.
DPO's exploratory reviews of the other seeds also gave no consistent upgrade.

## Completed, reviewable evidence

- [DPO experiment](dpo_review73_v1.md) and [per-case review](dpo_review73_validation_review.md).
- [Formal-input data pilot](synthetic_paired60_v1.md) and [per-case review](synthetic_paired60_validation_review.md).
- [Incoming-register ablation](synthetic_register_v1.md) and [per-case review](synthetic_register_validation_review.md).
- [Fresh evaluator-policy check](eval_policy_20261103.md): 6/6 reviewed policy examples matched.
- `experiments/v5_ablation_freeze_20261103.json` records decisions and source/result hashes.

Keep all existing model/data provenance and raw results. A null/negative result
is evidence about the tested recipe, not proof that preference learning or
synthetic supervision can never help. Do not repeatedly tune on these same 50
conversations until an improvement appears.

## Actual next work: main evaluation and deliverables

The main ten-persona/multi-system evaluation is **not complete**. The exporter’s
input check passed (50 validation queries, 190 bank items), but it loaded no
models and generated no end-to-end quality evidence.

1. Keep the existing selected supervised checkpoint/decoding for the baseline
   comparison; preserve V4, retrieval and hybrid identities and real providers.
2. Produce matched real validation outputs for the compared systems. Verify
   generator-only/hybrid execution and record actual fallbacks; do not label a
   retrieval fallback a successful hybrid.
3. Use fresh judge-run folders and the updated selectable-draft rubric. Six
   one-persona policy checks passed; pilot representative system replies before
   the main panel. Keep all persona settings/model digests matched. Ratings stay
   evaluation-only, separate from training labels and prior-rubric historical runs.
4. Finish tables, errors/qualitative cases and the demo proof. Freeze every system
   and judge configuration before the one final held-out test run. Do not open
   the test set simply because these training ablations are finished.

The rubric now permits plausible new progress, room, deadline and approval
answers as user-selected drafts, while rejecting explicit contradictions and
reversed roles. It does not certify invented details as true. Personal willingness
to send is simulated and conditional; no independent human feedback is claimed.
