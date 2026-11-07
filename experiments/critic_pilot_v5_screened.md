# Screened critic training pilot — 2 November 2026

Job **349291**. Rubric `selectable-v5-user-reviewed`, isolated screen `selectable-single-v3-user-reviewed`. Input is the first 20 of 380 existing training-only comparisons; this is not a random sample.

The six user-reviewed policy examples passed 6/6; the fresh assistant-authored transfer check matched candidate-acceptability expectations on 24/24 candidates. Neither is a held-out benchmark.

## Pilot result

- Kept preferences: **6/20**.
- Both bad: **7**; skipped without promoting an unjudged authored reference.
- Ranking changed across reply orders: **5**; skipped.
- Ties: **2**; skipped.
- Separate acceptance screening rejected no additional order-consistent winners in this pilot. It was applied to both candidates only when pairwise consensus selected a winner.

The assistant inspected all six kept preferences, including full context and both candidates. They favor a relevant printer follow-up, an offer to address a printing issue, a usable sign-up-link reply, a coherent schedule answer, a natural meeting acknowledgment and waiting for a late friend. No clear contradiction or reversed actor was found in the selected replies. This is assistant inspection, not independent human validation of the six labels.

| Comparison | Preferred draft |
| --- | --- |
| 1 | whats wrong with it ? |
| 8 | no worries, what do you want to print? |
| 10 | can, i send you the link |
| 11 | ya, tuesday got class |
| 13 | okay, I'll meet you there. |
| 18 | Okay, I'll wait for you. |

## Next step

Expand labeling to all 380 comparisons with the same rubric, settings and screening, in a new output. The full run repeats the initial 20 for one self-contained report. Inspect counts, failures and retained labels after completion before any DPO training. The pilot yield is not an accuracy estimate and should not be projected as a guaranteed full-run yield.

The first 20 are mostly STANDARD (13 STANDARD, 7 LOCAL), and were not stratified, so this pilot does not independently validate broad Singlish competence. Ties and skipped both-bad cases can discard useful information; no claim is made that all skipped candidates are objectively bad. Some acceptability explanations overstate redundancy as contradiction; assess the selected preference and underlying context rather than trusting explanations as facts.

Raw report: ignored `artifacts/critic_pilot_v5_screened.json`. No generator weights changed. 25 focused code tests passed; syntax/diff checks passed.

Full run **349304** started on `aloha`, output `artifacts/critic_full_v5_screened.json`. It labels 380 comparisons; it does not start training. Job 349291 completed in 5m 21s; transfer job 349274 completed in 4m 12s. No ongoing monitoring is scheduled.
