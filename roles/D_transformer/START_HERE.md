# Role D — Transformer experiment

## This week (Week 4)

Open `notebooks/D_mt5_kaggle_smoke_test.ipynb` on **Kaggle GPU** (or Colab). It must:

1. Load `google/mt5-small` locally (transformers + sentencepiece).
2. Run one forward pass.
3. Record CUDA memory and parameter count.
4. Save `mt5_smoke_report.json`.

Do not call any hosted generation API.

## Next

- Week 5: input format `STANDARD:` / `LOCAL:` prefixes using development data only.
- Week 6–7: small fine-tune, then fair comparison with V4 on the same dev set.
- If V5 is not clearly better, stop. V4 remains the system generator.
- Week 10–11: clean reproduction notebook.

## Friday artefact

The memory JSON from Kaggle, pasted into `experiments/logs/`.
