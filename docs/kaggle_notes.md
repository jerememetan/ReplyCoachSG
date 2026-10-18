# Kaggle notes for Role D

1. Create a notebook, GPU T4, internet on.
2. Upload `notebooks/D_mt5_kaggle_smoke_test.ipynb` or paste its cells.
3. Run all. Expected: `google/mt5-small` has on the order of 300M parameters; peak memory should fit a T4 at max_length 64.
4. Download `mt5_smoke_report.json` into `experiments/logs/`.
5. Fine-tuning later uses development JSONL only. Prefix inputs with `STANDARD` or `LOCAL` via `replycoach.generate.style_control.mt5_input`.
6. If validation on the same development set does not beat V4, stop. Do not stack V5 on V4.

## V5 stage 1: dialogue training on public pairs

mT5-small has only seen span corruption, so it first learns to reply on public pairs, then fine-tunes on development items.

1. Get Role C's `dailydialog_train_pairs.jsonl` and `dailydialog_validation_pairs.jsonl`, so V4 and V5 see the same DailyDialog.
2. After Role A adds CoSEM to `docs/licence_checklist.md`, download `corpus_COSEM_v5.zip` into `data/public/` and run:

   ```powershell
   python scripts/prepare_cosem_pairs.py --input data/public/corpus_COSEM_v5.zip --train-output data/public/cosem_train_pairs.jsonl --validation-output data/public/cosem_validation_pairs.jsonl
   ```

3. Upload the four files as a private Kaggle dataset and run `notebooks/D_mt5_pretrain_kaggle.ipynb`.
4. DailyDialog replies are tagged `STANDARD` and CoSEM replies `LOCAL` through `mt5_input`. The fine-tune loads `artifacts/mt5_v5_pretrain/best/`.
5. Keep full precision. mT5 gives NaN losses under fp16.
6. V5 validation loss is per subword and V4's is per word, so the two numbers cannot be compared. Compare generated replies on the development set.
7. On the SCIS GPU cluster, skip the notebook: copy the repo and the four pair files over, build `.venv` on the login node, load `google/mt5-small` once there so it is cached, then from the repo root run `sbatch --partition=... --account=... --qos=... scripts/v5_pretrain.sbatch smoke`, and the same without `smoke` for the full run. `myinfo` prints your partition, account and QOS.
8. No development or test item is read in this stage. Local install: `pip install -e ".[mt5]"`.

## V5 stage 2: LoRA fine-tune on development items

1. Each item in `data/dev/train.jsonl` gives two pairs, `STANDARD` → `replies.standard` and `LOCAL` → `replies.local`, in the same `mt5_input` format as stage 1. One adapter learns both styles; the tag picks the style.
2. `data/dev/validation.jsonl` picks the epoch and is the V4 vs V5 comparison set. `data/test/test.jsonl` is never read; the trainer refuses any item whose split is not `dev`.
3. Epoch 0 in the report is the stage 1 model alone on the validation items, the number fine-tuning has to beat. Training stops after `--patience` epochs without improvement.
4. On the cluster, after `pip install -e ".[mt5]"` (adds `peft`): `NAME=mt5_small_lora sbatch --partition=... --account=... --qos=... scripts/v5_finetune.sbatch`. `BASE=artifacts/mt5_base/best` uses the mT5-base stage 1 run; `--lora-rank 0 --learning-rate 1e-4` is the full fine-tune comparison.
5. `artifacts/<NAME>/best/` holds only the adapter; it names its base checkpoint, so keep the stage 1 `best/` folder in place.
