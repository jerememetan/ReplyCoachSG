# SEA-LION corrected abstention study V3

Status: completed on personal account `hm.teoh.2024`, NVIDIA L40S. Training358131 completed exit0 in10m59s; saved-checkpoint replay358145 completed exit0 in12s. Validation selected step288. The custom replay exactly matches all seven diagnostic outputs from training evaluation. No default was changed.

This implements selective abstention for missing required facts. A blank history alone does not trigger refusal: greetings, invitations, arithmetic, common knowledge and facts supplied in the latest message have answerable controls. The English model must generate “Information not available.” itself. Only that exact normalized output bypasses the existing Singlish rewrite; no input-based refusal or earlier context-rule layer is used.

The run resets to the original repair25 English LoRA, after correcting 144 speaker-relative synthetic questions from V1/V2. All original 790 examples, including 190 team-authored records, remain unchanged. The original base and Singlish adapter stay frozen. V1/V2 checkpoints remain historical diagnostics because 72 supported targets had potentially invalid speaker grounding.

Training: 1,152 unique examples per epoch: 790 original English examples, 120 missing-information examples (40 empty, 40 irrelevant, 40 pressure-to-guess), 120 supported counterparts and 122 self-contained positive controls. Missing-information supervision is 10.42%. Two epochs, 288 updates, effective batch 8, LR 1e-5; existing rank-8 LoRA updates 10,485,760 of 3,178,255,360 parameters. See `recipe.json` and `audit.json`.

Selection uses generated validation output: two layouts of 48 prior scenarios plus 32 self-contained factual controls. It maximizes the worse history-layout balanced abstention score, subject to each layout having <=10% exact over-refusals, lexical value matches no lower than baseline minus one, and no generation failures. These are diagnostics, not semantic accuracy. Step 0 remains eligible.

Evaluation freezes 24 new synthetic inputs (eight supported/missing scenario pairs plus eight self-contained cases), 20 reused old regression examples and seven existing user diagnostics. The latter guide interpretation but never checkpoint selection. No locked project test, independent human benchmark, API generation, source-checkpoint overwrite or automatic deployment change.

Custom inference uses the saved selected adapter, existing base and frozen Singlish adapter on the personal GPU account. The validated custom CLI is `run_sealion_abstention_custom_v3.py` in the remote study directory. Inputs use the original handoff schema (`item_id`, `context`, `speakers`, `message`); exact model-emitted abstention yields `needs_more_context=true`. Other drafts are not verified answers. Noncanonical uncertainty is flagged separately.

## Completed result

The model learned selective abstention on the constructed tasks, but an overall conversational improvement is not established. The eight factual scenarios each have supported/missing counterparts, plus eight self-contained controls (24rows total).

| Diagnostic on fresh24 synthetic rows | Original | Corrected selected288 |
|---|---:|---:|
| Exact standard abstention on8 missing-fact rows |0/8|7/8|
| Noncanonical-or-canonical uncertainty cues on missing rows |1/8|7/8|
| Exact false refusal on16 answerable rows |0/16|0/16|
| Expected value appears in13 factual answerable replies |9/13|12/13|

These are exact-output and lexical diagnostics, not semantic accuracy. The zero false refusals on this small new suite does not generalize to other inputs: validation selected288 still has2/33 false refusals in the canonical history view and1/33 in the alternate view (two views of the same48 underlying scenarios). The32 self-contained factual validation controls have0 exact refusals and24 lexical matches.

The actual user call with no supplied purpose now yields `Information not available.` in both styles, with `needs_more_context=true`. This is an internal missing-context signal; it is not a natural reply to send to the friend. More seriously, the known-purpose diagnostic, whose history explicitly states “our assignment deadline”, also yields that refusal; the original model correctly names the deadline. This is a regression. Both diagnostic inputs were excluded from training and selection, but had been inspected during earlier development.

Positive examples: missing agreed meeting time changes from an invented6PM to abstention;18:45 provided in the latest message is now answered;2plus3 remainsfive; lunch invitation no longer invents12:00. Failures: an unspecified key collector is still invented (“I am collecting the keys.”); budget15/ticket10 gets a deferral; old charger-direction, tomorrow/now timing and cross-style meaning problems remain. See `assessment.json`. No new Singlish adapter was trained; the original repair25 rewriter renders English except when exact abstention bypasses it.

Full51-row original/selected/final comparison: `results/review/paired_comparison.csv` (24new synthetic+20reused regressions+7reused diagnostics). Selected and final are the same checkpoint here, not two independent models. Snapshot selection used only the predefined validation criteria. The original English/base/Singlish checkpoints and original190 team-data hash remain preserved.

Verified results-only archive: `results_only_v1.tgz`,241075bytes, SHA256`c7a3c45f3022985374e203032bbd5a2e8ad0ae2aee69537755e3c19d653054b3`,39regularfiles, no weights. Extraction: `results/`. The interpretation in local `assessment.json` was added after downloading that immutable raw archive. Selected checkpoint stays at `~/ReplyCoachSG_sealion_abstention_v3/outputs/selective_abstention/selected`; step hashes are in `results/checkpoint_inventory.json`. Original790/team190 unchanged; no locked-test access, model promotion or new API generation.
