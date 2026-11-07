# Experimental input-register adapters — 3 November 2026

Status: **not selected as an upgrade**. Existing supervised V5 remains selected.
Full results: [input-register ablation](../../experiments/synthetic_register_v1.md).

- Architecture/base/LoRA: same mT5-base stage-1 checkpoint, rank 16, alpha 32,
  dropout 0.1, q/v targets as the matched formal-input experiment.
- Supervision: same 190 authored conversations/context pairs plus the same
  59 fictional conversations and unchanged STANDARD/LOCAL targets. Only history
  and incoming-message wording changes. Twenty-two incoming messages include
  Chinese characters; other changes include shorthand and colloquial syntax.
- Provenance: assistant-authored variants; Gemma reviewed original/new situations.
  58 approvals and one documented author correction of an erroneous possessive
  interpretation. This is not independent human/native-speaker review.
- Training: seeds 425/1/2, lr 3e-4, batch 16, 280 updates, 8-epoch cap, patience 2;
  no DPO/unlikelihood. All runs ended at the budget in partial epoch 7.
- Mean validation token loss: 2.5499 (formal augmentation 2.5308; control 2.5068).
- Primary assistant review: 14 register vs 19 control preferences; 35 ties,
  7 both-bad, 25 identical. Prior comparator exposure limits blinding.
- Selected checkpoint/default provider: unchanged. No test-set evaluation.
- Artifacts: `CS425G9:~/ReplyCoachSG/experiments/synthetic_register_v1/artifacts/register_s{425,1,2}/best`.

This limited style bundle did not establish improved reply quality. It does not
prove that all colloquial inputs, code-switching or synthetic supervision harm
generalization. The same validation conversations were reused during development;
only the primary seed has qualitative ratings. All source/review/model provenance
and the documented reviewer correction remain archived.
