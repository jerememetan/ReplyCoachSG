# Judge-selection experiments and validation pilot

Recorded 18 October 2026. Owner: Jereme / Role E.

## Purpose and decision

These experiments tested **which LLM to use as an evaluator**, not which LLM
should generate ReplyCoach replies. The judge rates an anonymous proposed next
reply for appropriateness, naturalness and requested style fit (each 1–5), plus
whether its fictional persona would send it.

We selected **local `gemma4:12b` through Ollama** as a practical comparative judge.
Earlier models showed contextual or style-scoring errors; Gemini's hosted quota
interrupted the panel. Gemma completed the full pilot and caught some important
contradictions, but it also made errors. This is not proof that Gemma is the most
accurate judge, a validated Singlish expert or equivalent to human reviewers.

## Experiments performed, 17–18 October

| Experiment | Saved evidence and observation | Decision / limitation |
| --- | --- | --- |
| Qwen2.5-Coder 7B | Completed 200 ratings. P01 gave an already-booked-court reply proposing another booking appropriateness 4 and would-send yes. | Valid JSON and completion did not establish contextual correctness. |
| Hermes3 8B, initial and revised prompts | Two completed 200-rating panels. Both missed the same court contradiction in the inspected P01 judgments. One penalised a STANDARD reply for not matching the peer's Singlish. | Separate style from context and requested style from persona habits; do not trust averages alone. |
| Hermes concise evaluator-only rubric | Six inspected replies, unchanged P01; schema-valid responses still rewarded the contradictory court reply with appropriateness 5. | Removing administrative prompt text alone did not fix judging. |
| Hermes single-case ablations | Removed persona, removed API schema constraint, or shortened rubric separately. All still accepted the court mismatch (appropriateness 4–5); a full-history plain-language check also missed it. | No tested single change explained or fixed this failure. One case is not a general error-rate estimate. |
| Hosted Gemini (`gemini-3.8-flash`, recorded identifier) | One-persona run saved 5/20 ratings; ten-persona run saved 16/200 before HTTP 429. | Incomplete quota-limited experiment, not a completed system comparison. Hosted integration was subsequently removed. |
| Qwen3.5 9B, thinking enabled | First diagnostic timed out at about 180 seconds. Retry with a 4,096-token context / 2,048-token output budget returned a truncated response after about 213 seconds. | No rating accepted from either incomplete response. Not practical evidence of reliable thinking-mode judging on this setup. |
| Qwen3.5 9B, thinking off and shorter persona | Three mismatch cases each. Thinking-off run rated all three appropriateness 2; shortening the persona improved one case but changed the printer case from 2 to 5. | Mixed effects. Shortening also changed wording, so length was not isolated. Full CSV personas were not replaced. |
| Neutral Qwen3.5 9B vs Gemma4 12B | Same six examples, no persona, thinking off: three problematic replies and three assistant-authored positive controls. Both rejected two mismatches and accepted all positive controls, but rated the printer mismatch appropriateness 5. | Useful small diagnostic, not independently reviewed gold or a benchmark proving Gemma superior. Proceed to the full Gemma persona pilot with limitations recorded. |

Prompts, settings and persona conditions changed during judge development. Do
not rank judge accuracy by comparing these runs' aggregate means or pool their
ratings. The positive controls were diagnostic only; they were not added to the
authored training dataset or presented as human-approved references.

## Completed Gemma pilot

Saved run: `run_20261018_135142_278527` under ignored `experiments/judging/`.

- Five shared **validation** conversations; no held-out test items.
- BM25 retrieval-only replies with M0 and M2 memory, both STANDARD and LOCAL.
- 20 candidate replies × 10 existing fictional personas = **200 ratings**.
- All 200 distinct candidate/persona pairs succeeded on their first attempt;
  no recorded failures or missing pairs. None returned `uncertain=true`.
- Sum of recorded per-judgment durations: **2,300.35 seconds (about 38.3 minutes)**,
  not a separately measured end-to-end pipeline wall time.
- Model digest: `6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b`.
- Thinking off; temperature 0; seed 425; context 4,096 tokens; output limit 2,048.

The following values are the saved pilot summary, not final evaluation results.
Scores are 1–5; send percentages are simulated willingness, not human approval.

| Setting | Style | Appropriateness | Naturalness | Style fit | Would send | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BM25 M0 | LOCAL | 3.12 | 4.60 | 5.00 | 36% | 50/50 |
| BM25 M0 | STANDARD | 2.64 | 4.98 | 5.00 | 36% | 50/50 |
| BM25 M2 | LOCAL | 3.94 | 4.88 | 4.94 | 68% | 50/50 |
| BM25 M2 | STANDARD | 3.20 | 4.84 | 5.00 | 56% | 50/50 |

M2 has higher saved appropriateness/send averages in this small pilot. Five
conversations do not establish a general improvement or statistical significance.
MiniLM, RNN and mT5 were **not** compared in this Gemma run.

## What inspecting individual judgments revealed

- **Caught contradiction:** C000003 says the court is already booked for seven,
  but the reply offers to book it once the time is confirmed. All ten personas
  gave appropriateness 1 and would-send false.
- **Caught unsupported referent:** C000002 discusses a difficult first lesson;
  the STANDARD reply is "Sure, you can use mine." Mean appropriateness was 1.2,
  with no would-send votes.
- **Still overrated a missed intent:** C000001's peer does not know how to operate
  the printer. "Can try the second floor one." does not explain operation and
  introduces an unsupported location, yet mean appropriateness was 4.2 and nine
  personas would send it. The neutral diagnostic also missed this issue.
- **Weak style discrimination:** 199/200 ratings assigned style fit 5. Inspect
  this ceiling effect rather than claiming the criterion is well calibrated.
  Earlier diagnostics also showed persona habits affecting requested-style scores.

Successful formatting, no uncertainty flags and agreement across personas do not
prove correctness. Ten personas share one model and are correlated simulated
perspectives, not ten independent human reviewers. Selection/prompt checks reused
known validation cases, so they cannot independently validate judge reliability.

## Handoff and remaining work

The local judging implementation, completed pilot and judge-selection decision
are available. Use the same model/digest, rubric, personas, settings and shared
conversations for each compared system. Manually spot-check additional validation
cases, ideally with an independent teammate, and retain/report disagreements.
Independent teammate checks are not established by these saved runs.

Main validation evaluation, real-generator/hybrid comparisons and the final
held-out evaluation remain separate work. Do not tune from the held-out test set
or turn these ratings into training labels. See the
[evaluation protocol](split_and_rating_protocol.md) for the full procedure.

## Evidence locations and reproduction

Raw runs remain ignored/local; this report intentionally commits no credentials,
absolute workstation paths, private mappings or raw conversation datasets.

- Early panels: `run_20261017_193355_050879` (Qwen),
  `run_20261017_201746_421528` and `run_20261017_205629_449086` (Hermes).
- Hermes diagnostics: `calibration_20261017_211116_880932` and
  `ablation_20261017_212235_913002`.
- Gemini: `run_20261017_215735_150486` (one persona) and
  `run_20261017_215900_177217` (quota-interrupted panel).
- Qwen3.5: `qwen35_diagnostic_20261018_105427_340999`,
  `qwen35_retry_20261018_105755_637056`, `qwen35_no_think_20261018_110253_985705`,
  `qwen35_short_persona_20261018_111455_661868`, and
  `qwen35_neutral_six_20261018_131238_337542`.
- Gemma diagnostic: `gemma4_neutral_six_20261018_134241_878742`.
- Full Gemma pilot: `run_20261018_135142_278527`, containing
  `packet/ratings/run.json`, `ratings.jsonl` and `summary.csv`.

Display the existing Gemma summary without model calls:

```powershell
.\.venv\Scripts\python.exe scripts/run_judging.py --run-dir experiments/judging/run_20261018_135142_278527 --summary-only
```

That command requires the local saved folder, which is not in Git. For a new run,
follow [README setup](../README.md#run-persona-judging--one-command). Current code
records its own fingerprints: do not silently resume/relabel the historical run
after prompt or source changes. Other teammates can reproduce the workflow, not
assume identical scores across changed software/model versions or hardware.
