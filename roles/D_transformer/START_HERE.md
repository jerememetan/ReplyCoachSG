# Role D — Transformer experiment (V5, mT5)

No hosted generation API. Train on `data/dev/train.jsonl`, validate on `data/dev/validation.jsonl`, never open `data/test/test.jsonl`.
Work branch: `role-d/v5-finetune`. Cluster commands need `--partition=project --account=cs425 --qos=cs425qos` (written `...` below).

## Pipeline

| Stage | Script | What it teaches | Status |
|---|---|---|---|
| 0 (optional) | `spans_v5.py`, `v5_spans.sbatch` | Read Singlish: span corruption on NUS SMS + CoSEM text | built, not run |
| 1 | `train_v5.py`, `v5_pretrain.sbatch` | Reply: DailyDialog (STANDARD) + CoSEM (LOCAL) pairs | done: small 3.466, base 3.345 |
| 2 | `finetune_v5.py`, `v5_finetune.sbatch` | Task and styles: LoRA on 190 dev items × 2 styles | done: base LoRA 2.511 (best), small LoRA 2.639, small full 2.621 |

Stage 2 problems seen on validation: LOCAL replies often not local (24/50 with a marker vs 35/50 in references), copying the context, mixing up who said what.

## To do

### 1. Augmentation runs (each about a minute; compare with 2.511)
- [ ] `rsync -av --exclude .venv --exclude artifacts ./ CS425G9@10.193.104.102:~/ReplyCoachSG/`
- [ ] `NAME=base_lora_ctx BASE=artifacts/mt5_base/best sbatch ... scripts/v5_finetune.sbatch --context-variants`
- [ ] `NAME=base_lora_reh BASE=artifacts/mt5_base/best sbatch ... scripts/v5_finetune.sbatch --rehearsal-pairs data/public/dailydialog_train_pairs.jsonl data/public/cosem_train_pairs.jsonl`
- [ ] `NAME=base_lora_both BASE=artifacts/mt5_base/best sbatch ... scripts/v5_finetune.sbatch --context-variants --rehearsal-pairs data/public/dailydialog_train_pairs.jsonl data/public/cosem_train_pairs.jsonl`

### 2. Stage 0 chain (compare stage 1 with 3.345 and stage 2 with 2.511)
- [ ] Check: `NAME=spans_check sbatch ... scripts/v5_spans.sbatch --train-limit 2000 --epochs 1 --eval-every 50`
- [ ] `NAME=mt5_base_spans sbatch ... scripts/v5_spans.sbatch` (roughly 30–60 min)
- [ ] `NAME=mt5_base_after_spans MODEL=artifacts/mt5_base_spans/best EPOCHS=5 sbatch ... scripts/v5_pretrain.sbatch` (~2 h; EPOCHS=5 matches the original base run)
- [ ] `NAME=base_spans_lora BASE=artifacts/mt5_base_after_spans/best sbatch ... scripts/v5_finetune.sbatch` plus the best flags from step 1

### 3. Read the results
- [ ] On the laptop: `scp 'CS425G9@10.193.104.102:~/ReplyCoachSG/artifacts/*.json' artifacts/`
- [ ] For each run: epoch 0 vs best loss, STANDARD vs LOCAL loss, when it over-fit, LOCAL marker rate, how often it copies the input
- [ ] Pick the final V5 checkpoint

### 4. Connect and compare
- [ ] Fill in `load_v5()` in `src/replycoach/generate/providers.py`: base + adapter + tokenizer, `mt5_input` format, Candidates for both styles with `source="V5"` and a score in [0, 1] (e.g. exp of mean token log-prob)
- [ ] Decide where V5 runs for the demo (the laptop cannot load mT5-base comfortably)
- [ ] Export dev outputs: `python scripts/export_dev_outputs.py --generator v5 --output <file>`
- [ ] Compare with V4 on the same 50 validation items once Role C connects `load_v4()`; if V5 is not clearly better, V4 stays

### 5. Write-up
- [ ] Stage 0/1/2 design and why (self-supervised next-turn prediction vs span corruption, one tagged adapter vs two, LoRA vs full)
- [ ] Results tables, augmentation and stage 0 ablations, compute cost (GPU, minutes, peak memory)
- [ ] Limitations: 190 items, validation used for epoch choice and the V4 comparison, dated NUS SMS language, facts the model cannot know
- [ ] Ask Role A to add CoSEM, NUS SMS and mT5-base to `docs/licence_checklist.md`; cite Gonzales et al. (2023) and Chen & Kan (2013)
- [ ] Clean reproduction notebook (weeks 10–11)

### Set aside
- Llama-SEA-LION + LoRA: possible extra experiment later (8B, needs the 48 GB GPUs; the demo cannot run it on a laptop).

Commands and data steps: `docs/kaggle_notes.md`.
