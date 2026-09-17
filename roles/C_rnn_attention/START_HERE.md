# Role C — RNN and attention generator

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
