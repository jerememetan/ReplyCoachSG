# SEA-LION latest-turn repair experiments

The three new English-adapter training runs did not establish a reliable improvement over the original repair25. The original checkpoint remains intact. An optional hybrid pipeline now supplies explicit-context English drafts for supported cases, followed by the original neural Singlish rewriter and a conservative rewrite check.

## Training work

- Personal account: hm.teoh.2024. All new training used one L40S; no project-account GPU jobs.
- Frozen source: exported SEA-LION repair25;3.178B frozen backbone,10,485,760 trainable parameters in the English LoRA. Original Singlish adapter stayed frozen.
-192 assistant-authored targeted English dialogues:96 same-history/different-latest-question pairs.790 existing English replay examples.16 separately written synthetic validation examples and50 reused ordinary validation examples.
- Three120-update runs: supervised repair; the matched supervised repair plus role/fact/latest-question margin loss; and supervised repair with explicit Me/history/latest-incoming serialization. Same starting English adapter; LR2e-5. Per step one ordinary example and three targeted positives, plus the matched contrast graph. Three snapshots per run:40/80/120, nine newly trained snapshots in total.
- Original190 team records unchanged. The exact user call conversation was a diagnostic only, excluded from training. No project locked-test access. No generation API or runtime Gemma.

## What the results support

Both canonical training arms and the explicit-layout arm retainedstep0 under the frozen ordinary-loss retention rule. Targeted validation loss sometimes improved while ordinary validation loss increased. We also generated all16 validation English replies from every stored snapshot, with shuffled model labels. Assistant review found12 passes/1 uncertain/3 failures for the original; no trained snapshot exceeded12 passes, and the tied snapshots had4 failures. The reviewer authored the synthetic labels; this is not independent human accuracy.

Forty separately written challenge scenarios were initially frozen before their respective runs, plus20 reused regression conversations and the user diagnostic. Once inspected and used to develop the optional rules, these became development material. Ten additional assistant-authored smoke cases were later generated and inspected; those are development material too. None establishes fresh human generalization.

A same-device audit found identical official-export/study prompt tokens, adapter tensors and outputs on A40. Original outputs differ between A40 and L40S despite identical checkpoints. All paired training comparisons used L40S. This is a numerical/backend variation, not a source checkpoint change.

## Optional context policy

The rule module receives only the conversation, latest message and original English reply. It cannot read target labels. Supported patterns include explicit numeric budgets, earliest times, busy days, borrowing direction, own location, request clarification and call-purpose follow-ups. It uses the stated purpose where supported; otherwise it can propose a neutral text-first action. It does not infer a private reason for calling.

Rule-generated English is labelled `english_provenance=context_rule:...`. It is not claimed as a newly learned neural reply. The original Singlish adapter rewrites that anchor. When limited surface checks reject the rewrite, LOCAL contains the English draft with `singlish_status=english_fallback`. No rule match does not establish correctness; unsupported or complicated contexts can still fail.

Final policy revisionv4:71 development/diagnostic conversations;22 rule matches;20 English strings changed;49 unmatched pairs preserved exactly;11 LOCAL outputs use English fallback.31 regression tests pass. Counts are technical diagnostics, not semantic accuracy. Earlier policy revisions/results are retained, including a corrected unnecessary rewrite and a corrected rewrite introducing a past conversation. Final v4 adds only a CPU check to actual job358077 generations; it does not retrain or regenerate them.

## Your call example

| Output | Exact text |
|---|---|
| Original English on matched L40S | Could you call me now? |
| Original Singlish on matched L40S | Not sure what u want to talk about |
| Context-rule English | I'll send you a message explaining what I wanted to discuss. |
| Actual neural Singlish rewrite | Okay, I send you a message explaining what I wanted to discuss. |

This is a neutral proposed next action, not an answer claiming to know the unstated call topic. The standalone English model still fails this example.

## Review files

- [Final original-versus-policy comparison](results_v3/review/context_policy_comparison.csv): full contexts, incoming messages, original and final replies, rules and provenance.
- [Training comparisons](results_v3/review/turn_repair_comparison.csv).
- [Input-layout comparisons](results_v3/review/layout_comparison.csv).
- [All-snapshot generated validation scores](results_v3/review/generated_validation_scores.json).
- [Final raw policy results](results_v3/context_policy_v4/results.json).
- [Checkpoint inventory](results_v3/checkpoint_inventory.json): remote locations and SHA256 for nine snapshots and three selected copies. No weights downloaded.

## Custom inference

The optional pipeline is available on the personal account at:

`~/ReplyCoachSG_sealion_turn_repair_v1/context_policy_v4`

Inside an allocated compatible GPU job, using the existing personal Python environment:

```bash
module load Python/3.13.1-GCCcore-13.3.0
cd "$HOME/ReplyCoachSG_sealion_turn_repair_v1/context_policy_v4"
"$HOME/ReplyCoachSG_focus_v1/.venv/bin/python" -I run_sealion_context_policy_v1.py \
  --inputs /absolute/path/custom_inputs.json \
  --output /absolute/path/fresh_outputs.json
```

Omit `--cached-baselines` for custom inference. That option belongs to the frozen study replay. The output path must be new. Input is a nonempty JSON list with unique item_id, context strings, matching user/peer/other speakers, and latest incoming message. The original export still runs independently; the main app and handoff defaults were not replaced.

## Preservation

Training job358042 completed5m52s; layout job358056 completed4m08s. Other completed GPU checks:358044,358047,358059,358060,358061,358074,358077. Failed launch358068 ran0seconds because its launcher named a nonexistent script; corrected before inference, logs retained. Personal queue was empty after the final jobs.

Final results-only archive:results_only_v3.tgz,446646bytes, SHA256b8f83e48b4abbece50510709999c3a95bce65954e5a485fc2cac5b6132814f8a,132 verified regular files, no model weights. Source models and original team data remain unchanged. No Git push or checkpoint promotion was performed.
