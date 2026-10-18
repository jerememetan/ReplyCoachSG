# Public data (not the dual-style task)

Put DailyDialog downloads here, then run:

```powershell
python scripts/prepare_dailydialog_pairs.py --input data/public/dailydialog.jsonl --output data/public/dailydialog_pairs.jsonl
```

For V5 only, put `corpus_COSEM_v5.zip` from https://github.com/wdwgonzales/CoSEM here, then run:

```powershell
python scripts/prepare_cosem_pairs.py --input data/public/corpus_COSEM_v5.zip --train-output data/public/cosem_train_pairs.jsonl --validation-output data/public/cosem_validation_pairs.jsonl
```

CoSEM is real chat text with automatic scrubbing only. Keep it and its pairs out of Git, and check any line for personal details before quoting it. Cite Gonzales, Hiramoto, Leimgruber & Lim (2023).

Licence: CC BY-NC-SA 4.0 for both corpora. See `docs/licence_checklist.md`. These pairs train V3/V4 for generic replies. They do **not** replace team-authored STANDARD/LOCAL items.
