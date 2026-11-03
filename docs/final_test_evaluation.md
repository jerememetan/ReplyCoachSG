# Frozen 60-item final evaluation

The user authorized the held-out run on 2 November 2026. This is separate from
the completed 50-conversation validation evaluation. Do not use its outputs or
ratings for checkpoint choice, training labels, prompt adjustment or tuning.

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

## Confirmed progress on 3 November

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
