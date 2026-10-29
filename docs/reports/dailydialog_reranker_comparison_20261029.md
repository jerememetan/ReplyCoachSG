# DailyDialog Qwen reply-selection comparison — 29 October 2026

**Conclusion:** the new adapter has not established a useful overall improvement over previous Qwen or V5. Standard selection is effectively unchanged versus previous Qwen; Local improves slightly in point estimates. V5-only still has the highest mean appropriateness in both styles. Keep the new adapter experimental and preserve the current demo default.

**Subsequent demo direction (29 October):** the user plans to use this checkpoint
as the final demo **V6 (Qwen + DailyDialog)** reranker. This is a product/demo
choice alongside the mixed evaluation results above. The subsequent integration
sets this as the All models default and retains the MiniLM baselines as options;
the comparison itself did not change the app default.

## Matched evaluation

Completed a fresh replay with the imported final adapter on the same 50 validation conversations, 400 saved proposals and 4,000 existing Gemma AI-persona ratings as the earlier experiments. Each conversation supplies Standard and Local proposals from BM25, MiniLM, V4 and V5. Qwen ranks those replies; it is not generating its own replies. V5-only always selects the saved V5 reply. The earlier Qwen and V5 measurements are reused from the matching frozen packet, rather than generating new outputs.

The packet/export/rating hashes, query/bank identities, loaded adapter tensor hash, prompt, base revision, CPU precision, 512-token budget and replay script identity were checked. No new judge calls or training were performed. The 60 held-out project test items remain untouched. The demo and model defaults were not changed.

## Primary results

Appropriateness is the mean existing AI rating on a 1–5 scale. “Would send” is the existing AI-persona yes rate, not a human-survey result or accuracy percentage.

| Selector | Standard appropriateness | Local appropriateness | Standard would-send | Local would-send |
| --- | ---: | ---: | ---: | ---: |
| Previous Qwen + reviewed-only LoRA | 3.640 | 3.946 | 60.8% | 70.0% |
| DailyDialog → reviewed Qwen | 3.634 | 3.988 | 59.8% | 73.4% |
| V5-only | 3.666 | 4.110 | 61.2% | 76.8% |
| Raw Qwen (context) | 3.578 | 4.044 | 58.8% | 72.4% |
| Reviewed MiniLM (context) | 3.508 | 3.930 | 55.4% | 71.0% |

## Paired differences and uncertainty

20,000 paired bootstrap resamples, seed 425, resampling whole conversations. Both styles and the ten correlated persona ratings remain together within their conversation. Intervals are descriptive development uncertainty conditional on this fixed output pool and judge; they are not held-out human confidence guarantees.

| Comparison | Style | Appropriateness difference | Descriptive 95% interval |
| --- | --- | ---: | --- |
| New − Previous Qwen + reviewed-only LoRA | STANDARD | -0.006 | [-0.088, +0.088] |
| New − Previous Qwen + reviewed-only LoRA | LOCAL | +0.042 | [-0.152, +0.232] |
| New − Previous Qwen + reviewed-only LoRA | BOTH_STYLES | +0.018 | [-0.088, +0.125] |
| New − V5-only | STANDARD | -0.032 | [-0.414, +0.358] |
| New − V5-only | LOCAL | -0.122 | [-0.356, +0.104] |
| New − V5-only | BOTH_STYLES | -0.077 | [-0.319, +0.164] |

All four style-specific appropriateness intervals against the requested baselines include zero. Averaging both styles within each conversation gives only +0.018 versus previous Qwen and −0.077 versus V5. No reliable overall gain is established. This does not prove the systems are equivalent.

## Which replies changed?

Against previous Qwen, only **10 of 100 reply texts changed**: 3 Standard and 7 Local. Existing appropriateness ratings are higher for five changed choices, lower for four and equal for one. This is a rating comparison, not five independently verified corrections.

| Item / style | Incoming message | Previous Qwen selection | New selection | AI appropriateness change |
| --- | --- | --- | --- | ---: |
| ITEM-C-023 / STANDARD | want me to jio you for the night market thing? | Sure! What time should we meet? | sure i'll meet you at the entrance | -1.1 |
| ITEM-D-007 / STANDARD | then what u wanna drink later? | what would you like? I will get it for you | I'd like to drink a bottle of wine. | -1.0 |
| ITEM-E-032 / STANDARD | sorry ah but that one cannot, someone book liao | Hurry! You don't want to be late for your first class | yes, the lecturer is inside alr, but class hasnt started yet | +1.8 |
| ITEM-A-017 / LOCAL | Eh i double check with you, we are meeting at 7pm right? | yes, I triple check with you that I am going | Yep, 7pm is fine. | +2.0 |
| ITEM-A-038 / LOCAL | i might be ten mins late leh | Okay, want to confirm the time and place? | yah, ofc | +1.8 |
| ITEM-B-029 / LOCAL | which room is our class in? | at seminar room. why eh | Our class is in the second floor. | +0.0 |
| ITEM-C-001 / LOCAL | i'll be at lks around 8, want to meet there? | Okay, want to confirm the time and place? | okay, i'll meet you there | +2.2 |
| ITEM-C-028 / LOCAL | paiseh, i ordered first because i was too hungry | yes i was waiting for you | havent leh, i was waiting for you | +0.8 |
| ITEM-D-009 / LOCAL | neh the gaussian elim qn | gna finish soon. whats ur ans for the first question | eh i see | -2.3 |
| ITEM-E-007 / LOCAL | onz, tmr meet na li | Okay, want to confirm the time and place? | okay, tmr got work | -2.4 |

Qualitative inspection of the saved text (assistant analysis, not new human labels):

- **Clearer time confirmation, ITEM-A-017 Local:** the peer asks whether the meeting is at 7pm. New Qwen selects “Yep, 7pm is fine.” instead of the awkward “yes, I triple check with you that I am going”. It selects the V5 proposal here.
- **Clearer meeting acceptance, ITEM-C-001 Local:** with the peer offering to meet at LKS around 8, the new selection is “okay, i'll meet you there” instead of asking to confirm the already supplied time and place. Again, it selects V5.
- **Regression, ITEM-E-007 Local:** after agreeing on Chinatown, the peer asks “onz, tmr meet na li”. The new selection “okay, tmr got work” does not answer where to meet; the previous selection asks to confirm time/place.
- **Persistent context/role failures versus V5:** for ITEM-A-046 Standard, the peer already suggests 2pm, but the new selector asks “what time should we meet?” instead of V5's acceptance. For ITEM-E-009 Standard, the peer asks to borrow Your charger, but the selector chooses a reply promising to return it, apparently reversing the borrower/lender roles.

## Important AI-rating limitation

ITEM-B-057 Standard illustrates why these averages cannot be treated as definitive quality labels:

> Peer: my phone is almost out of battery
>
> You: I brought my charger
>
> Incoming Peer: can I borrow your charger?

The new selector chooses “Ok , I will return once I have charged my phone”. The frozen AI ratings give this **5.0/5 appropriateness and 100% would-send**, despite its apparent borrower/lender reversal. V5's “Sure, I'm going to take a look at it.” is also a weak answer and scores 2.0. A +3.0 AI-rating difference here is not evidence that the new reply is correct.

Other nominal improvements remain questionable: ITEM-E-032 still introduces a lecturer/class-start claim when the peer is declining a room change; ITEM-C-028 still fails to directly acknowledge the peer's apology for ordering first. These are reasons to retain independent qualitative/human checks and avoid using judge scores as training labels.

## Repeated-speaker histories

Only five validation conversations contain an adjacent same-speaker pair in the supplied history plus incoming Peer message: ITEM-D-033, ITEM-D-037, ITEM-D-038, ITEM-E-027 and ITEM-E-028. **All ten Standard/Local choices are unchanged versus previous Qwen.** Mean appropriateness stays 2.46 Standard / 3.50 Local; V5-only gives 3.58 / 4.08 on this small subgroup. This run therefore provides no observed improvement for the user's message-burst concern, but five conversations are insufficient for a robust subgroup conclusion.

## Duplicate-text sensitivity

The 400 source records contain 381 distinct conversation/style/reply texts and 19 duplicate-text groups. The primary analysis preserves the existing deterministic anonymous-ID tie policy. A sensitivity analysis averages all frozen ratings for identical text within a conversation/style, ensuring identical answers have identical evaluation values regardless of source.

It yields 3.631 / 3.988 for the new adapter, 3.637 / 3.946 for previous Qwen, and 3.666 / 4.110 for V5. The new-minus-previous appropriateness differences remain −0.006 / +0.042, and the conclusion against V5 is unchanged. No ratings were created, and this aggregation is evaluation-only.

## Interpretation and decision

- The lower selected project-pair loss (0.349055 → 0.341845) did not translate into a demonstrated overall reply-selection gain. Random weak-negative classification and choosing among plausible generated replies are different evaluation demands.
- Extra DailyDialog adaptation changed relatively few decisions and includes both useful corrections and regressions. The run does not isolate the effects of data source, extra update budget and numerical recovery, and does not prove that a larger corpus or more epochs alone would solve role/time errors.
- Keep this adapter as a separately identifiable experiment. V5-only retains higher observed means; no automatic demo promotion or model replacement is justified by these results.
- If refining further, obtain reviewed role/time/commitment negatives from training conversations. Do not turn these validation examples or their AI ratings into training labels.
- The existing AI judge has documented semantic failures, repeated validation has influenced development, and the persona ratings are correlated. A small targeted independent human check would add evidence that these automatic comparisons cannot supply.

## Evidence and reproduction

Ignored private evidence under `artifacts/reranker_qwen/`:

- `replay_dd_20261029_01/result.json` and `summary.csv`: fresh 50-conversation replay.
- `comparison_dd_20261029.json`: paired intervals, all metrics, sensitivity and subgroup summaries.
- `selection_changes_dd_20261029.json` and UTF-8-BOM `.csv`: complete per-conversation comparisons against previous Qwen and V5.
- `compare_dd_20261029.py`: evaluation-only aggregation with data/hash/mean checks.
- `replay_dd_with_progress_20261029.py`: progress wrapper delegating unchanged scoring and replay code.
- `dd_result_20261029/`: verified training artifacts and final adapter.

Observed CPU ranking time was 616.78 seconds, with 48.99 seconds loading/integrity checks. Prior Qwen replay was approximately 481 seconds. Hardware contention was not controlled, so these are observed run times, not a latency-performance claim. Selected-query truncation flags: 0 of 100.

```powershell
.\.venv\Scripts\python.exe scripts/replay_shared_ranker.py `
  --packet experiments/judging/four_systems_20261026_01/main_packet `
  --qwen-model artifacts/reranker_qwen/pretrained `
  --qwen-adapter artifacts/reranker_qwen/dd_result_20261029/stage2_reviewed/best `
  --out-dir artifacts/reranker_qwen/NEW_REPLAY_DIRECTORY
```

Use a fresh output directory. Base/model prompt hashes and training/validation identities must pass. The result files record the exact adapter, packet, rating and script identities. No raw private data or weights are added to Git by this report.
