# Model card — V5 (mT5-base + LoRA)

Role D owns this card. Numbers come from the run reports in `artifacts/*.json` (gitignored; copied back from the cluster).
Fields marked **TODO** are not measured yet.

## Card summary (`docs/licence_checklist.md`)

| Field | Value |
| --- | --- |
| Parameter count | 584,170,752 at inference (mT5-base 582,401,280 + LoRA 1,769,472); 1,769,472 trained in stage 2 (0.30%) |
| Training time | about 3.6 h in total: stage 0 37 min, stage 1 3.0 h, stage 2 1.5 min |
| GPU type | NVIDIA GeForce RTX 3090 (24 GB), SMU `crimson` SLURM cluster; not Kaggle/Colab |
| Peak memory | 18.1 GB (stage 1); 14.6 GB (stage 0); 8.5 GB (stage 2) |
| Beat V4 on the same development set? | **TODO**: no V4 vs V5 comparison on `data/dev/validation.jsonl` yet |

## Model details

| | |
| --- | --- |
| Version | V5, the transformer generator of ReplyCoach |
| Checkpoint | `artifacts/spk1_ul025_s425/best` (LoRA adapter, tokenizer and `v5_input.json`); shared with the team in `v5_ul025_weights.tar` |
| Base | `google/mt5-base` (Apache 2.0) → stage 0 `mt5_base_spans` → stage 1 `mt5_base_after_spans_spk` (full fine-tunes) |
| Adapter | LoRA rank 16, alpha 32, dropout 0.1, on attention `q` and `v` |
| Input | `STANDARD:` / `LOCAL:` style tag, then the conversation with each turn marked `Friend:` or `You:`, then the message; up to 128 source tokens |
| Output | One reply per style, up to 48 tokens; beam search (4 beams, no repeated 3-grams) |
| Score | exp(mean token log-probability) of the reply, in [0, 1] |
| Serving | `load_v5()` in `src/replycoach/generate/providers.py`; the default checkpoint, or set `V5_CHECKPOINT` |
| Hugging Face card | `spk1_ul025_s425_README.md` (copy over the blank `README.md` in the checkpoint folder) |

## Training

All stages: seed 425, one RTX 3090. Time is training plus validation, without model loading, from the report's last `seconds` entry.
Memory is `torch.cuda.max_memory_allocated`, so `nvidia-smi` shows a little more.

| Stage | Script | Data | Settings | Best validation loss | Time | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| 0: span corruption | `spans_v5.py` | 96,295 text chunks (CoSEM 89,638, NUS SMS 6,657) | 2 epochs, lr 3e-4, batch 32, 15% noise, mean span 3 | 2.459 at step 6,020 (the last step) | 37 min | 14.6 GB |
| 1: reply pairs | `train_v5.py` | 156,052 pairs: CoSEM sample 80,000 (`LOCAL`), DailyDialog 76,052 (`STANDARD`) | 5 epochs, lr 5e-4, warmup 1,000, batch 32, speaker labels | 3.335 at step 19,508 (CoSEM 4.339, DailyDialog 2.330) | 3.0 h | 18.1 GB |
| 2: team items (LoRA) | `finetune_v5.py` | 190 `data/dev/train.jsonl` items × 2 styles = 380 pairs, + 180 context pairs | up to 30 epochs, patience 5, lr 1e-3, batch 16; unlikelihood weight 0.25 on 89 echo replies from 34 items | 2.494 at epoch 5 (`STANDARD` 2.157, `LOCAL` 2.831) | 1.5 min | 8.5 GB |

Stage 2 was also run with seeds 1 and 2 (2.477, 2.488). Seed 425 is the checkpoint used here.

### Reproduce

Cluster flags (`--partition=project --account=cs425 --qos=cs425qos`) are written `...`.

```bash
NAME=mt5_base_spans sbatch ... scripts/v5_spans.sbatch
NAME=mt5_base_after_spans_spk MODEL=artifacts/mt5_base_spans/best EPOCHS=5 sbatch ... scripts/v5_pretrain.sbatch full --speaker-labels
NAME=spk1_ul025 SEEDS="425 1 2" BASE=artifacts/mt5_base_after_spans_spk/best sbatch ... scripts/v5_finetune.sbatch \
  --speaker-labels --context-pairs --negatives artifacts/candidates_train_spk1.json --unlikelihood-weight 0.25
```

## Evaluation

All on the 50 items in `data/dev/validation.jsonl`. `data/test/test.jsonl` has not been opened.
Validation loss was also used to pick the epoch and the run, so these numbers are optimistic.

| Comparison | Result |
| --- | --- |
| Validation loss vs the same setup without unlikelihood (`spk1_lora_cp`), mean of 3 seeds | 2.486 vs 2.482 |
| Blind A/B by Role D vs `spk1_lora_cp` (`artifacts/ab_ul025.json`, 81 questions; 19 identical pairs skipped) | this model 24, `spk1_lora_cp` 15, tie 15, both bad 27; sign test p = 0.20 |
| vs V4 on the same 50 items | **TODO**: export both with `scripts/export_dev_outputs.py`, then judge or rate blind |

V4's reported loss is on DailyDialog with a word-level vocabulary, so it cannot be compared with these mT5 token losses.

A later run, `full_ul025_s425`, trained stage 1 on all of CoSEM: validation loss 2.409, and 25–21 against this model in blind rating (p = 0.66).

## Intended use

- Suggest a `STANDARD` and a `LOCAL` (Singlish) reply to a peer's chat message, for the CS425 ReplyCoach demo.
- Runs locally; no hosted generation API.
- Not for real conversations about health, money, safety or anything where a wrong reply matters.

## Data, licences and privacy

| Resource | Use | Licence / terms |
| --- | --- | --- |
| `google/mt5-base` | base model | Apache 2.0; **TODO** Role A to add to `docs/licence_checklist.md` (only mT5-small is listed) |
| DailyDialog | stage 1 `STANDARD` pairs | CC BY-NC-SA 4.0 |
| CoSEM (Gonzales, Hiramoto, Leimgruber & Lim, 2023) | stage 0 text, stage 1 `LOCAL` pairs | **TODO** Role A to confirm and add |
| NUS SMS Corpus (Chen & Kan, 2013) | stage 0 text | no formal licence; the authors ask for citation and an email; **TODO** Role A to add |
| Team-authored items (`data-v1` split) | stage 2 | team-owned; draft pending team review |

CoSEM and NUS SMS are real messages with automatic scrubbing only. They stay out of Git, and any quoted line is checked for personal details.
No LLM-generated data or labels were used for training.

## Limitations

- **About a third of replies are bad** (27 of 81 rated "both bad"). The common failures:
  - generic "okay / no worries, I'll ___" replies that don't answer the message;
  - mixing up who does what (e.g. "dabao your food" → "I'll dabao for you");
  - not answering direct questions (where, which room).
- Only 190 training items. Stage 2 over-fits within about 10 epochs.
- CoSEM pairs are noisy: in a hand-check of 40 pairs, 13 replies did not answer the message.
- About 450 NUS SMS anonymisation tags (`<DECIMAL>` etc.) were left in the stage 0 text, and NUS SMS language is dated (2010s).
- The model cannot know facts about the user's life (schedules, places, plans) and may invent them.
