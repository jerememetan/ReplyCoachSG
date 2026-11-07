# Instruction-chat baseline and reward diagnostics

This follows the completed volume experiment. No online RL or default-model replacement is part of this run.

## Baseline

Generate one STANDARD and one LOCAL draft for each of the same 50 development validation conversations, with the cached `gemma4:12b` model. Pin its digest and record the system prompt, deterministic decoding settings, runtimes and source hashes. Do not include reference replies in generation prompts. Compare its outputs with `control_long_s425` from the volume experiment using anonymized assistant inspection.

Gemma differs from mT5 in size, family, pretraining, quantization and decoding. Results therefore measure a practical capability baseline, not the isolated causal effect of instruction tuning. Gemma also supplies the proposed reward scorer; its own judgment of its own replies will not establish a winning model. The assistant reviewer knows the project and authored prior synthetic corrections, so masking identities does not create an independent human evaluation.

## New diagnostic

`data/reward_diagnostics/challenge_v2.jsonl` freezes 80 assistant-authored draft labels across 20 fictional scenario clusters, balanced acceptable/unacceptable. Five scenarios each concern explicit constraints, actor/ownership roles, valid choices/refusals/clarifications, and plausible unknown details. Gold labels, group names, rationales and IDs are excluded from scorer prompts. No diagnostic example enters training, no test file is read, and the six user policy examples are not counted as fresh validation.

Version 2 was frozen before scoring. A subsequent provenance check found that the attempted draft-text revisions had not applied: v1 and v2 contain the same draft texts, with different IDs and rationale metadata. The original snapshots/results preserve the earlier inaccurate description; this note corrects that metadata claim. Both versions remain available. Version 2 hash: `ff5573ee22f9f69ee7ac283a539d554672eee7dcc5c023800855ef2d598dd234`.

The proposed scorer returns acceptable (+1), unacceptable (-1), or uncertain (no reward; abstain). This is a prompted classifier, not a trained reward model or PPO value critic. Uncertainty is not silently converted into a negative training reward.

Frozen diagnostic gate: balanced accuracy at least 90%; at least 90% rejection of role negatives and constraint negatives separately; at least 90% acceptance of valid-choice and new-detail positives separately; abstentions at most 10%, counted as classification failures. These are engineering thresholds, not statistical reliability guarantees.

This diagnostic contains constructed contrasts, including some obvious errors. Passing it is **necessary but insufficient** to start RL. Inspect agreement on actual actor outputs next, including disputed cases and gaming of the rubric. Expected labels are assistant-authored, not human gold; the 80 drafts are correlated within 20 scenarios. Do not report them as 80 independent human judgments or tune the prompt repeatedly against the same labels.

## Execution

Frozen source/label recipe: `experiments/rl_readiness_v1_recipe.json`. Isolated remote snapshot: `~/ReplyCoachSG/experiments/rl_readiness_v1`. Runner: `scripts/v5_rl_readiness.sbatch`. Seven tests passed for label isolation, verdict/reward consistency, abstention handling and per-category gating; shell syntax check passed.

## Next decision

Finish the baseline reply comparison and inspect the scorer's failure cases. If it fails, document the failure and defer RL. If it passes, audit its scores on actual model outputs before a small supervised-versus-RL comparison. No outcome is assumed in advance.


## Baseline result

Job 350937 generated all 100 baseline replies. Frozen Gemma digest: `6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b`. Anonymized one-seed assistant comparison against long-trained mT5: 54 prefer chat baseline, two prefer mT5, 42 ties, two both bad, zero identical. These are 100 style comparisons on 50 reused conversations, not 100 independent human judgments. The same `na li` meeting-location question was inadequately answered in both styles by both models.

Assistant cross-style inspection flagged 14 of 50 pairs for possible semantic drift. Severity varies: clear differences include room 302 versus not knowing/checking the room, different drink preferences, and alternative shows versus alternative cinemas. Other flags concern narrower follow-up requests or missing details. The 14 flags must not be interpreted as equally severe failures or independent gold labels. Separate per-style generation has no mechanism enforcing identical intent. The baseline result supports further exploration of model/prompt capability but does not justify an automatic default replacement.

## Actual-output reward check

A second diagnostic was frozen before scorer outcomes: `data/reward_diagnostics/actor_outputs_v1.jsonl`, 80 actual mT5 drafts (40 acceptable, 40 unacceptable according to prior assistant inspection), clustered in 38 development conversations. The set is selected and balanced, not representative of model error rates, and expectations are fallible assistant labels. All label/source IDs remain hidden from the scorer. Job 350938 depends on successful completion of job 350937 and uses a separate frozen snapshot. Its pass threshold is >=90% balanced agreement and <=10% abstention; disagreements still require inspection.


## Constructed reward diagnostic result

Job 350937 completed in 5m47s, exit zero. Scorer agrees with 74/80 assistant-authored labels (92.5% balanced agreement), accepts 38/40 valid drafts and rejects 36/40 invalid drafts. No abstentions. **Fails the frozen gate** because it rejects only 8/10 role-reversal negatives, below the required 9/10. Do not waive that category failure because aggregate agreement exceeds 90%.

The failures are material: it accepts the user saying they are on the train when the user is explicitly waiting at the station; accepts reversed tablet ownership; accepts collection at five when the desk closes at four; and rejects two valid proposals asking the peer to collect before four on the user's behalf. It also accepts an incoherent booth/flyer reply. All ten valid plausible-new-detail examples are accepted, so the scorer is following that portion of user policy in this diagnostic. No reward weights were trained and no actor RL update has occurred.


### Actual-output run recovery

Job 350938 failed after 52 completed ratings because a response hit the configured generation limit; no full-set score is claimed from that partial run. The failed report is preserved as `artifacts/rl_readiness_v1/actor_reward_failed.json`. Recovery uses a fresh snapshot/output and imports only the unchanged prefix after verifying source SHA256, model/settings, prompt, reward mapping, gate and row identity/content. Truncated or malformed outputs are recorded as abstentions with raw failure evidence and no reward; transport errors still fail the job. No response-limit increase, prompt edit, relabeling or threshold change is used to obtain a pass. Eight tests cover the scorer contract and preservation of truncated-output evidence.


## Actual-output result and decision

Resume job 350941 completed successfully in 2m09s. All 80 actual drafts have recorded outcomes: 56/80 agree with the frozen assistant expectations (70% balanced agreement), including 31/40 expected-acceptable and 25/40 expected-unacceptable. One truncated response is an abstention with no reward. The actual-output gate fails. This is agreement with selected assistant labels, **not 70% human accuracy**.

Review of all 24 disagreements found ten clear scorer errors, five prior reference labels judged too strict, eight ambiguous reference cases and one abstention. This follow-up is itself assistant judgment, not independent human adjudication. Raw labels, scores and thresholds remain unchanged; no post-hoc corrected pass score is computed. For example, prior labels too harshly rejected a busy-course acknowledgement and a plausible offer to send a bidding link. Clear scorer failures include treating a request for a later slot as satisfied by an earlier slot and rejecting a legitimate question asking where an unspecified bus stop is. The raw earlier volume-comparison assistant counts should therefore remain exploratory judgments, not validated absolute-quality labels.

**Decision:** do not run online RL using this scorer. The larger chat model is a promising practical baseline, but independent generation produces semantic differences between styles and remains imperfect on a code-switched meeting-location question. Next actor work should enforce a shared intent across STANDARD/LOCAL drafts and evaluate that explicitly. Reward work needs a clearer adequacy policy and independent adjudication of disputed examples before another frozen validation. Default mT5 checkpoint remains unchanged; no actor weights or reward-model weights were updated. No jobs remain from this experiment.
