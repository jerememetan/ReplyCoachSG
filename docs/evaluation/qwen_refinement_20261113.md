# Qwen reranker refinement with SEA-LION and Jev

## What we tested

After the presentation, we tested whether SEA-LION could supply extra supervision for the Qwen3-Reranker-0.6B adapter. An initial pilot used 14 pairs contrasting a user-revised reply with its original SEA-LION draft. It showed a small, mixed change on the development replay. A follow-up using eight reviewer-selected comparisons produced identical replay scores to its matched training control.

The automated experiment generated SEA-LION replies for 247 synthetic conversations across two rounds. For each conversation, it looked for an edit to a single detail, such as the time, day, place, or quantity, that the existing Qwen ranked above the starting reply. The starting reply was either SEA-LION's Local output or a reference written for the scenario. Jev saw both replies in randomized order and scored each for contextual fit. A pair entered training when Jev gave the lower-ranked reply at least 0.80 and the higher-ranked reply at most 0.20. Other pairs were skipped. Each round saved 12 accepted pairs for optional blind human audit.

| Round | Jev classified | Accepted | SEA-LION starting replies among accepted | Scenario references among accepted | Unsure | Both fit |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 95 | 78 | 54 | 24 | 15 | 2 |
| 2 | 90 | 79 | 47 | 32 | 9 | 2 |

Jev's reported cost was USD 0.004033. Each round's two arms started from the same original adapter and shared 120 original training preferences and 28 earlier human preferences. The candidate added that round's Jev-labeled preferences, and the matched control added an equal number of original preferences. Each arm trained for two epochs with a learning rate of 1e-5, batch size 2, gradient accumulation of 8, and 58 optimizer updates. The arms used 452 labeled replies each in Round 1 and 454 in Round 2. Validation loss selected the checkpoint within each arm.

## Results

The promotion gate required the candidate to beat both the original and the matched control on eight saved human preferences. It also required each style's replay to stay within 0.02 appropriateness points and one percentage point of would-send of the original. The ordinary replay used 400 historical proposals for 50 development conversations and 4,000 saved Gemma persona ratings.

| Checkpoint | Human preferences matched | Style | Appropriateness / 5 | Naturalness / 5 | Style fit / 5 | Would send |
|---|---:|---|---:|---:|---:|---:|
| Original Qwen | 7/8 | Local | 3.988 | 4.930 | 4.996 | 73.4% |
| Matched control, both rounds | 6/8 | Local | 4.046 | 4.896 | 4.980 | 74.2% |
| Jev-trained candidate, both rounds | 5/8 | Local | 3.978 | 4.888 | 4.980 | 72.2% |

Both rounds produced the same aggregate results. Standard replay scores were identical across checkpoints at 3.634 appropriateness, 4.926 naturalness, 5.000 style fit and 59.8% would-send. The candidates fell short on the human check and dropped 1.2 percentage points on Local would-send. The run stopped after two consecutive failures and the original checkpoint stayed in the demo.

| Case | Original Qwen selected | Jev-trained candidate selected in Round 1 | Human preference |
|---|---|---|---|
| The peer asks to move a reading group from Tuesday to Wednesday. | “Wednesday works for me, let's go with that then.” | “Sure, Friday works for me.” | Wednesday |
| The peer asks if they can still borrow a power bank after previously asking to borrow it in an hour. | “Can, use it in one hour then return me ah.” | “sorry ah but i still need it leh, I can only return by next week” | The one-hour reply |

The original also made factual errors during mining. When asked to label a package on Tuesday before Friday collection, it preferred “I'll label it on Thursday before the courier collects it” over SEA-LION's otherwise identical Tuesday reply. Jev selected Tuesday, so this pair entered training.

The human check covered eight decisions across six synthetic conversations reused from an earlier experiment. Those decisions were excluded from both arms' training in this run. The Gemma ratings were simulated and the replay ranked historical proposals. The results support retaining the original adapter under this promotion gate. They do not isolate the cause of the regression or measure the final demo's V7 v2 Local refinement directly. In the current demo, Standard is direct SEA-LION and Qwen ranks the original Local reply against V7 wording variants when available.

## Context replay with the original checkpoint

We kept the Qwen checkpoint and candidate replies fixed and changed only the prior turns in its query. The comparison used default M2 memory versus all available nonblank turns in the latest eight-turn window. No scoring input was truncated at the model's token limit.

| Replay | M2 | Full recent history |
|---|---:|---:|
| Correct on 12 constructed cases that rely on earlier context | 6/12 | 10/12 |
| Correct on four controls with the key detail in the latest message | 4/4 | 4/4 |
| Selections on the frozen 50-conversation replay | 100 picks | Same 100 picks |

Full history fixed five of the constructed context cases and flipped one correct choice to an incorrect one. When the user asked to move a 7pm meetup one hour later, M2 dropped the original time and Qwen chose 7pm. With the earlier turn restored, Qwen chose 8pm. In another case, the user asked to move a 4pm pickup half an hour earlier. Full history made Qwen switch from 3:30pm to 4:30pm. It also kept choosing the south entrance even when the earlier turn said north.

On the historical replay, full history changed only 15 of 381 distinct query–reply inputs and all Gemma metrics stayed identical. The constructed cases deliberately exercise references to older turns, and the historical replay uses shared-ranking proposals from an earlier system. Neither measures the effect of changing live V7 v2 rankings. The evidence supports testing better retention of the turn that sets the time, place or commitment on fresh Local examples.

The local script and the saved per-candidate Qwen scores are in `output/qwen_refinement_loop/context_replay.py` and `output/qwen_refinement_loop/context_replay.json`.

## Targeted memory and the Standard reply decision

We tested a rule that keeps M2 and restores at most one older turn when a request refers to a time, place, quantity, payment or commitment. On the same 12 constructed cases, M2 got 6 right, targeted memory got 7, and full recent history got 10. Targeted memory corrected the 7pm-to-8pm reschedule and the $14 transfer, but it also introduced the 4:30pm error when the request said half an hour earlier. All three scored 4/4 on the explicit-detail controls. These constructed examples were reused to diagnose the heuristic.

We separately replayed the frozen 60-conversation SEA-LION comparison. For Standard, Qwen ranked the same BM25, MiniLM, V4 and SEA-LION proposals under each memory mode. For Local, it ranked the saved SEA-LION originals and V7 v2 wording variants. The published V6 Standard metrics and all 60 saved V7 v2 Local choices were reproduced under M2. No query was truncated and the comparison reused the existing Gemma grades.

| Standard policy | Appropriateness / 5 | Naturalness / 5 | Style fit / 5 | Would send |
|---|---:|---:|---:|---:|
| Direct SEA-LION | 4.555 | 4.975 | 5.000 | 88.0% |
| Qwen, all three memory modes | 3.995 | 4.940 | 5.000 | 71.5% |

Qwen replaced SEA-LION in 18 of 60 Standard cases. Its appropriateness difference versus direct SEA-LION was −0.560, with a paired 95% bootstrap interval of [−0.905, −0.257]. For all 60 conversations, M2 already retained all available history, so targeted memory and full history produced identical queries and identical picks. Local refinement was active in 18 of 60 cases. Neither memory change altered any Local pick.

This replay explains why direct SEA-LION remains the Standard default. It cannot estimate the effect of additional memory on longer conversations. A useful promotion test for Standard would use fresh conversations with relevant older turns and hold candidate replies fixed, then require better paired appropriateness than direct SEA-LION without a drop in would-send. Local should be evaluated separately on fresh V7 variants where memory actually changes the query.

The local scoring script and per-candidate outputs are in `output/qwen_refinement_loop/targeted_memory_replay.py` and `output/qwen_refinement_loop/targeted_memory_replay.json`.

## Standard replay on longer conversations

We ran the fixed Standard candidates from BM25, MiniLM, V4 and SEA-LION on 20 team-authored diagnostic conversations with six prior turns each. The candidates were generated once using M2. Qwen then ranked those same candidates with M2 and with all six turns. Three Gemma 12B personas graded the selected replies blind with all six turns, for 75 unique reply–persona grades after reusing identical replies within each conversation.

| Standard policy | Appropriateness / 5 | Naturalness / 5 | Style fit / 5 | Would send |
|---|---:|---:|---:|---:|
| Direct SEA-LION | 4.30 | 5.00 | 5.00 | 81.7% |
| Qwen with M2 | 3.75 | 4.95 | 5.00 | 66.7% |
| Qwen with full history | 3.80 | 4.95 | 5.00 | 66.7% |

Full history changed one Standard pick. When the peer said “sorry, took the wrong bus”, Qwen switched from “Yes, I'll meet you at the campus gate” to “No worries. Just let me know when you arrive.” The mean appropriateness gain versus M2 was +0.05, with a paired 95% bootstrap interval of [0.00, +0.15]. Would-send was unchanged. Full-history Qwen still trailed direct SEA-LION by 0.50 appropriateness points and 15.0 percentage points on simulated would-send. The diagnostic set and simulated graders limit generalization. The demo continues to use direct SEA-LION for Standard.

Local receipts, including the rankings and grading records, are in `output/qwen_refinement_loop/long_standard/`.

## Next experiment

1. **Check explicit facts before ranking.** Compare a version of the original Qwen that filters clear contradictions in the latest request, such as Wednesday becoming Friday. Apply the filter only when the requested detail and the reply's corresponding detail can both be identified. Let Qwen rank when the check is uncertain. A retrospective rules check agreed with 21 of 23 constructed cases and abstained on two. It used authored metadata about which fact to check, so detecting that fact in unseen conversations still needs testing.
2. **Train on the choices the demo actually makes.** Collect fresh SEA-LION Local originals and V7 variants, including awkward slang, changed meaning, natural wording and cases where both options fit or neither fits. Give Jev the clear factual pairs. Use a small blind human sample for Singlish style and borderline choices. Deduplicate within each conversation, cap repeated templates, and balance meaningful style and context failures before scaling.
3. **Measure the filter and the training change separately.** Freeze a fresh human-reviewed set grouped by conversation and template before training. Compare original Qwen, original Qwen with the fact check, and an equal-budget adapter trained on the revised pairs. Report paired preference accuracy, explicit fact error rate, Local style fit and would-send. Use the frozen Gemma replay as a secondary regression check.

The local research workspace retains the per-round training receipts, mined pairs, Jev responses, and audit sheets under `output/qwen_refinement_loop/`.

## Proposed larger fine-tuning study

The automated pilot added 78 Local comparisons in Round 1 and 79 in Round 2. Each round started from the original checkpoint. Although SEA-LION generated both styles, the experiment added no new Standard comparisons and did not use V7 v2 wording variants in training. Its candidate arm also had 114 Standard training rows versus 190 in the matched control. The pilot therefore cannot answer whether broader, style-balanced supervision from the final demo pipeline improves V6.

1. **Generate training candidates from the deployed pipeline.** Start with several hundred new conversations across intents and lengths, holding out whole conversation families before generation. For Standard, save the direct SEA-LION reply and the BM25, MiniLM and V4 alternatives that Qwen would be asked to rank. For Light Local, save the SEA-LION original and all three V7 v2 variants whenever the pool can form. Save the current Qwen picks as baseline predictions. Record how often Local falls back to the original, and report the number of usable comparisons by style and intent. The prior 60-case run had active V7 variants in 18 cases.
2. **Assign labels independently of Qwen.** Rate the candidate texts blind. Use Gemma's existing four criteria, with Jev available to screen clear context errors. Resolve or skip uncertain examples and audit a sample of the automatically labeled pairs. Use context fit as the first priority and naturalness and style fit to distinguish replies that preserve the same meaning. Track would-send separately. Exclude duplicates and ties from preference training, and keep provenance for each pair.
3. **Fine-tune and compare with a style-matched control.** Start both arms from the same Qwen adapter and mix in the same original supervision. Add separately labeled Standard and Local pairs to the treatment. Match each arm's counts by style and optimizer budget. Select the checkpoint on validation conversations. Report the counts of unique conversations and accepted pairs along with the training settings.
4. **Evaluate the selected checkpoint on held-out outputs.** Keep the candidate pools and memory fixed across old and new Qwen. For Standard, compare both checkpoints with direct SEA-LION. For Local, compare old and new Qwen on the same SEA-LION and V7 v2 pool, and report results both across all conversations and where the pool forms. Use blinded Gemma 12B ratings and report paired differences for appropriateness, naturalness, style fit and simulated would-send. Keep the existing demo choices until those comparisons justify a change.

The larger run freezes 309 synthetic conversations from 30 authored scenario families before candidate generation, with 226 training, 41 validation and 42 test conversations. Each family appears in only one split. Jev screens each distinct alternative against the SEA-LION original on training and validation conversations. Clear contextual preferences enter the label pool. A bounded subset of unresolved pairs, prioritizing Local wording comparisons, goes to a separate Gemma pass. Ties and uncertain Gemma ratings are excluded. The held-out evaluation uses three Gemma 12B personas and reports conversation-level paired intervals. These are synthetic scenarios and simulated judgments. The test split spans four scenario families.

### Completed larger study

All 309 candidate pools completed. Jev screened 812 pairs for USD 0.017962 and returned 283 confident preferences, all Standard. The Gemma follow-up used 177 unique candidate grades and recovered seven Local training preferences and three Local validation preferences. After limiting training to one preference per conversation and style and matching the original-data control, the treatment used 125 new Standard preferences from Jev and seven new Local preferences from Gemma. Each arm used 524 labeled reply rows, 270 validation rows, and 99 optimizer updates over three epochs. The selected epoch was three for the treatment and one for the control. We verified that held-out conversation IDs were absent from both training arms and validation.

| Style | Policy | Appropriateness / 5 | Naturalness / 5 | Style fit / 5 | Would send |
|---|---|---:|---:|---:|---:|
| Standard | Direct SEA-LION | 4.286 | 4.968 | 5.000 | 81.0% |
| Standard | Original Qwen or matched control | 4.214 | 4.968 | 5.000 | 78.6% |
| Standard | Fine-tuned Qwen | 4.286 | 4.968 | 5.000 | 81.0% |
| Local | Direct SEA-LION | 4.429 | 4.690 | 3.540 | 70.6% |
| Local | Original Qwen | 4.421 | 4.683 | 3.603 | 71.4% |
| Local | Matched control | 4.429 | 4.683 | 3.603 | 71.4% |
| Local | Fine-tuned Qwen | 4.429 | 4.683 | 3.643 | 71.4% |

The test used 276 unique blinded Gemma grades across 42 conversations. Fine-tuning changed one Standard selection relative to original Qwen. For “Want to meet before it closes, or do this tomorrow?”, the candidate chose “Tomorrow works for me.” The original and control chose “Yes, I'll meet you at the campus gate.” Relative to original Qwen, Standard appropriateness rose by 0.071 (paired 95% interval [0.000, 0.214]) and would-send by 2.38 percentage points ([0.00, 7.14]). The candidate tied direct SEA-LION on all four metrics. Their choices differed on one conversation.

The candidate changed three Local selections relative to original Qwen. Local style fit increased by 0.040 ([0.000, 0.103]), with the same would-send rate and naturalness. Among the 18 conversations where V7 supplied a choice, Local style fit rose from 3.426 to 3.519 and would-send stayed at 77.8%. These intervals resample conversations within four synthetic test families, so they give limited evidence about new scenarios. The original demo checkpoint was retained. The Local label bottleneck was contextual screening and the number of clear style preferences, despite generating both reply styles.

Per-case selections, training receipts and metric intervals are under `output/qwen_refinement_loop/final_study/`.

### Standard comparison on the presentation's 60 conversations

We replayed the original, fine-tuned and matched-control V6 adapters on the same frozen BM25, MiniLM, V4 and SEA-LION Standard candidates using M2. All original V6 picks matched the published run. We reused the original ten Gemma persona grades for each selected reply. The historical conversations had already been inspected and this replay is reported separately from the 42 held-out synthetic conversations.

| Policy | Appropriateness / 5 | Naturalness / 5 | Style fit / 5 | Would send |
|---|---:|---:|---:|---:|
| Original V6 | 3.995 | 4.940 | 5.000 | 71.5% |
| Matched training control | 3.978 | 4.940 | 5.000 | 71.3% |
| Fine-tuned V6 | 4.135 | 4.922 | 5.000 | 75.2% |
| Direct SEA-LION | 4.555 | 4.975 | 5.000 | 88.0% |

Fine-tuning changed seven of the 60 picks. Against original V6, appropriateness rose by 0.140 (paired 95% bootstrap interval [0.000, 0.337]) and would-send rose by 3.67 percentage points ([0.00, 8.67]). Naturalness fell by 0.018 and style fit stayed level. Direct SEA-LION still led fine-tuned V6 by 0.420 appropriateness points and 12.83 percentage points of would-send, supporting the existing Standard demo choice.

The local scoring script and per-case scores are in `output/qwen_refinement_loop/presentation_standard_replay.py` and `output/qwen_refinement_loop/presentation_standard_replay.json`.

### Suggested final-report wording

We investigated whether SEA-LION-generated replies could improve the V6 Qwen reranker through distillation-style fine-tuning. We paired SEA-LION replies with competing candidates and used Jev and Gemma to label reply preferences. On the same 60-conversation evaluation, the strongest fine-tuned checkpoint increased Standard appropriateness from 3.995 to 4.135 out of 5 and simulated would-send from 71.5% to 75.2%. Further fine-tuning yielded no additional gain on that evaluation, and direct SEA-LION retained the highest scores for Standard replies. See the [continuation study](qwen_standard_continuation_20261113.md).
