# Kaggle notes for Role D

1. Create a notebook, GPU T4, internet on.
2. Upload `notebooks/D_mt5_kaggle_smoke_test.ipynb` or paste its cells.
3. Run all. Expected: `google/mt5-small` has on the order of 300M parameters; peak memory should fit a T4 at max_length 64.
4. Download `mt5_smoke_report.json` into `experiments/logs/`.
5. Fine-tuning later uses development JSONL only. Prefix inputs with `STANDARD` or `LOCAL` via `replycoach.generate.style_control.mt5_input`.
6. If validation on the same development set does not beat V4, stop. Do not stack V5 on V4.
