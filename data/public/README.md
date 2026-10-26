# Public data (not the dual-style task)

Put DailyDialog downloads here, then run:

```powershell
python scripts/prepare_dailydialog_pairs.py --input data/public/dailydialog.jsonl --output data/public/dailydialog_pairs.jsonl
```

For V5 only, put `corpus_COSEM_v5.zip` from https://github.com/wdwgonzales/CoSEM here, then run:

```powershell
python scripts/prepare_cosem_pairs.py --input data/public/corpus_COSEM_v5.zip --train-output data/public/cosem_train_pairs.jsonl --validation-output data/public/cosem_validation_pairs.jsonl
```

That keeps 80,000 training pairs, about the size of DailyDialog. For every usable pair (about 275,000) with the same validation file, run:

```powershell
python scripts/prepare_cosem_pairs.py --input data/public/corpus_COSEM_v5.zip --train-output data/public/cosem_full_train_pairs.jsonl --held-out data/public/cosem_validation_pairs.jsonl --max-train-pairs 300000
```

CoSEM is real chat text with automatic scrubbing only. Keep it and its pairs out of Git, and check any line for personal details before quoting it. Cite Gonzales, Hiramoto, Leimgruber & Lim (2023).

For V5 stage 0 only, put `smsCorpus_en_xml_2015.03.09_all.zip` from https://github.com/kite1988/nus-sms-corpus here and run `scripts/prepare_singlish_text.py` (see `docs/kaggle_notes.md`). The NUS SMS Corpus is real SMS text; keep it and its chunks out of Git, check any line for personal details before quoting it, and cite Chen & Kan (2013). Its repository asks users to cite the paper and email the authors; it states no formal licence.

Licence: CC BY-NC-SA 4.0 for both corpora. See `docs/licence_checklist.md`. These pairs train V3/V4 for generic replies. They do **not** replace team-authored STANDARD/LOCAL items.
