# Frozen 60-item final evaluation

The user authorized the held-out run on 2 November 2026. This is separate from
the completed 50-conversation validation evaluation. Do not use its outputs or
ratings for checkpoint choice, training labels, prompt adjustment or tuning.

## Completed findings — 3 November 2026

The run completed at **21:41 SGT on 3 November**. All **4,800 unique source
candidate/persona judgments succeeded**, with zero remaining errors. The final
`result.json` records `completed: true`, 60 test conversations and 4,800 source
ratings. All ten system/style summary groups contain 600 ratings (60 conversations
× ten personas). V6's 120 selections reuse source ratings rather than adding new
judge calls. Coverage, selection mappings and all four summary metrics were
independently recomputed from the saved journal and checked against the receipt.

| System | Standard appropriateness /5 | Local appropriateness /5 | Standard would-send | Local would-send |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 2.700 | 3.542 | 32.5% | 57.3% |
| MiniLM | 2.942 | 3.635 | 40.2% | 60.5% |
| V4 | 2.255 | 2.537 | 18.5% | 26.7% |
| V5 | 3.615 | 3.672 | 59.3% | 60.0% |
| V6 (Qwen + DailyDialog) | 3.657 | 3.748 | 60.0% | 61.7% |

**Interpretation:** V6 has the highest observed appropriateness means in both
styles. Its advantage over V5 is small: **+0.042 Standard and +0.077 Local**.
Paired uncertainty analysis is still pending; these point estimates do not
establish a statistically reliable improvement. Would-send percentages are
simulated AI-persona votes, not measured human acceptance. Ratings from the ten
personas are correlated; they do not turn 60 conversations into 600 independent
test conversations per style. The judge's previously documented semantic errors
and style ceiling remain limitations.

This evaluates the frozen **V6 selection pipeline**, not the later V7 local
refinement. Keep these held-out results separate from the earlier 50-conversation
validation comparisons. Do not use these findings to tune the evaluated models.

Private evidence in `experiments/judging/final_test_20261102_01/`:
`freeze.json`, `outputs.jsonl`, `v6_selections.json`, `result.json`,
`packet/judge_inputs.jsonl`, `packet/PRIVATE_mapping.jsonl`, and
`packet/ratings/{run.json,ratings.jsonl,summary.csv}`. Full naturalness and
style-fit means are preserved in `result.json`. Raw test text, replies, weights
and persona journals remain outside Git; only this aggregate summary is published.

## Fixed design

- 60 test conversations; only the 190 training items form the retrieval bank.
- BM25, MiniLM, V4 and V5 each supply Standard and Local proposals, using M2.
- Final V6 is Qwen + DailyDialog, selecting one source proposal per style.
- 480 anonymous source proposals × the same ten local Gemma personas = 4,800
  ratings. V6 reuses ratings of its selected source text, with the lowest
  anonymous candidate ID breaking identical-text source ties, as frozen before
  judging. This does not create additional independent evidence.
- The original local judge digest, rubric, personas and generation settings are
  retained. These are AI-persona judgments, not a human survey.
- Source/data hashes and the 27 supplied model-file hashes are recorded in
  `experiments/judging/final_test_20261102_01/freeze.json`. Resume refuses changed
  frozen inputs or configuration. Existing development-data guards remain intact.

## Historical restart on 3 November

The later V7 demo refinement is **not part of this frozen evaluation**. The
original source/data hashes remain in `freeze.json`; the current V7 checkout
differs from that source. A later wrapper resume must not bypass that rejection
by rewriting the freeze or regenerating test replies with the newer pipeline.
The saved anonymous packet and judge's unchanged resume mechanism preserve
completed ratings independently of later demo edits. Committing files does not
change the already-running evaluator's loaded code or its saved outputs.

All 60 model comparisons were exported. Before resuming, the rating journal
contained 310 successful candidate/persona judgments and no errors. No process
was active. The same run was restarted with `--resume`; do not start a second
final-test experiment or overwrite these files.

```powershell
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
.\.venv\Scripts\python.exe scripts/run_final_test.py --out experiments/judging/final_test_20261102_01 --resume
```

Run only when no other copy is active. Keep the PC awake and Ollama available.
Saved replies and successful ratings are reused. Raw evidence remains ignored.
Progress logs for the background restart are under
`artifacts/reranker_qwen/final_test_20261103.stdout.log` and `.stderr.log`.

Completion requires 4,800 successful unique judgments and the runner's
`result.json`, including source-system and V6 summaries. Intermediate summaries
are provisional; do not report them as final model quality. The run is not marked
complete merely because a background process was launched.
