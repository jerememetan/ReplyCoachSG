# Frozen 60-item final evaluation

## Current-pipeline regrade: 8 November 2026

**Completed:** 7,200/7,200 cells, zero failures; cluster job 358114 exited 0:0
after 6h 45m 33s, finishing at 22:26 Singapore time. See the
[verified six-system findings](evaluation/current_pipeline_20261108.md) and
[aggregate evidence](evaluation/current_pipeline_20261108.json).
SEA-LION alone has the highest appropriateness means. V7 improves Local style
fit and simulated would-send over V6, while leaving Standard unchanged.

The separately authorized re-evaluation uses the same 60 test conversations and
six systems: BM25 (V0/V1 normalization), MiniLM (V2), V4 attention RNN,
V5 **SEA-LION 3B**, V6 Qwen + DailyDialog selection, and V7 Local refinement.
V3 is excluded because its supplied saved replies cover validation only; trained
weights were not received for these test cases.

```powershell
.\.venv\Scripts\python.exe scripts/run_final_test.py --current-pipeline --out experiments/judging/final_current_20261108_01
```

This profile requires an allocated CUDA GPU for generation and the exact cached
Gemma judge. Cluster job **358114** was launched on `aloha` (RTX 3090) on
8 November. Its isolated workspace is `~/ReplyCoachSG_final_cluster_20261108_01`.
The local ignored `output/current_final_cluster_job.json` records its paths;
`progress.json` in the run folder is authoritative for progress. The laptop
mirror copies evidence only and does not run inference.

Fresh replies and the **current** rubric are frozen in a new run; old-rubric
grades are never imported. The 720 candidate rows (6 x 60 x 2) yield **7,200
nominal candidate/persona cells**. Identical complete conversation/style/reply
tasks can reuse the same persona grade within this new run. Reports retain
provenance counts. Ten personas grade appropriateness, naturalness, style fit,
would-send, problem category and uncertainty. The final report includes paired
conversation-level bootstrap intervals for V6 versus V5 and V7 versus V6.

V6 and V7 share Standard replies; V7 changes only Local selection. V7 uses the
existing three-variation-plus-original algorithm with a fixed per-case seed
(`425:<item_id>`) for reproducibility. No generators, rankers, rubric criteria,
or model weights are trained or changed. Missing model outputs fail the run.

These test conversations have already been inspected. Present this as a
re-evaluation after validation-based SEA-LION selection, with correlated AI
personas rather than independent human ratings. Do not tune from these scores
or describe this as a fresh unseen test. Historical runs below remain intact.

## Historical frozen evaluation

The interactive demo's V5 slot changed to SEA-LION 3B after the separate
[8 November transformer validation comparison](evaluation/transformer_selection_20261108.md).
The frozen results documented here used **original mT5**, not SEA-LION. The
legacy final-test runner still explicitly requests the original `v5` provider.
The separate `--current-pipeline` profile above now records the SEA-LION + Qwen
+ V7 re-evaluation; do not relabel these historical mT5 scores as SEA-LION.

## Include V7 in a new final report

Run the existing entry point with a separate output directory:

```powershell
.\.venv\Scripts\python.exe scripts/run_final_test.py --include-v7 --out experiments/judging/final_test_v7_01
```

Append `--resume` to continue that directory. This mode reads the completed
4,800-judgment baseline in `final_test_20261102_01`, applies V7 to its saved
V6 Local winners, and records all four candidates and the chosen reply before
judging. The evaluation uses a fixed sampling seed. It reuses Standard grades
and exact Local grades from the same conversation, and sends only new final
Local replies to the same Gemma model and ten personas.

The new `result.json` reports all six systems, the actual new judgment count,
and paired V7 minus V6 Local means. Those differences are descriptive AI-judge
results. The sections below describe the original completed baseline.

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
