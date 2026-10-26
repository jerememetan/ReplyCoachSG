# Role C — RNN and attention generator

## Current status — 20 October 2026

The five-epoch Colab V4 checkpoint was recovered from Drive and verified.
V3/V4 task fine-tuning, the shared Candidate provider and matched 50-query
development exports are implemented. See [HANDOFF.md](HANDOFF.md) and
[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for actual results, reproduction
commands and quality limitations. V5 comparison remains pending Role D.
The week-based instructions below are historical.

## This week (Week 4)

```powershell
python -m replycoach.generate.train_v3
```

This loads the 20 pilot pairs, trains **one CPU batch**, and writes `artifacts/rnn_v3_smoke.pt` plus `artifacts/rnn_one_batch.json`.

Code: `src/replycoach/generate/seq2seq.py` (V3 decoder and V4 attention decoder), `src/replycoach/generate/train_v3.py`.

## Next

- Week 5: train V3 on public DailyDialog pairs (`scripts/prepare_dailydialog_pairs.py`). Do not train on the held-out benchmark.
- Week 6: `--attention` for V4 and record validation loss.
- Week 7: tune V4, save best checkpoint.
- Week 8: emit `Candidate` objects in the shared format.

## Friday artefact

The JSON report with `loss` and `checkpoint`.
