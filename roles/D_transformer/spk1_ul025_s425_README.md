---
base_model: artifacts/mt5_base_after_spans_spk/best
library_name: peft
language:
- en
license: other
pipeline_tag: text2text-generation
tags:
- base_model:adapter:artifacts/mt5_base_after_spans_spk/best
- lora
- transformers
- mt5
- singlish
- reply-suggestion
---

# Model Card for ReplyCoach V5 (`spk1_ul025_s425`)

A LoRA adapter on a further-trained mT5-base that suggests two replies to a friend's chat message: one in standard English (`STANDARD`) and one in Singapore colloquial English (`LOCAL`, Singlish).
It is V5, the transformer generator of ReplyCoach, a CS425 course project.

## Model Details

### Model Description

- **Developed by:** CS425 Group 9 (ReplyCoachSG team), Role D
- **Funded by [optional]:** Not applicable (course project)
- **Shared by [optional]:** CS425 Group 9
- **Model type:** Encoder-decoder transformer (mT5-base) with a LoRA adapter, for conditional generation
- **Language(s) (NLP):** English and Singapore colloquial English (Singlish)
- **License:** Not decided. The base model is Apache 2.0, but the training data includes DailyDialog (CC BY-NC-SA 4.0), CoSEM and the NUS SMS Corpus, whose terms Role A has not yet confirmed. Do not redistribute.
- **Finetuned from model [optional]:** `google/mt5-base`, through two full fine-tunes (stage 0 → `artifacts/mt5_base_spans/best`, stage 1 → `artifacts/mt5_base_after_spans_spk/best`); this adapter is stage 2

### Model Sources [optional]

- **Repository:** ReplyCoachSG (private course repository), branch `role-d/v5-finetune`
- **Paper [optional]:** Not applicable
- **Demo [optional]:** The ReplyCoach demo (`replycoach` CLI)

## Uses

### Direct Use

Given a short chat history and the friend's latest message, generate one `STANDARD` and one `LOCAL` reply suggestion for the user to pick from or edit.
Everything runs locally; no hosted generation API is called.

### Downstream Use [optional]

In ReplyCoach, V5's candidates go to a shared reranker together with retrieved replies (V2) and, when connected, the RNN generator (V4).
Each candidate carries a score in [0, 1]: exp(mean token log-probability) of the reply under the model.

### Out-of-Scope Use

- Sending replies automatically without a person reading them.
- Conversations about health, money, safety, legal matters or anything else where a wrong reply matters.
- Statements of fact about the user's life (schedules, places, plans): the model does not know them and may invent them.
- Languages other than English and Singlish. mT5 is multilingual, but this adapter was not trained or checked on other languages.

## Bias, Risks, and Limitations

- **About a third of replies are poor.** In a blind rating of 81 validation questions, 27 were rated "both bad" for this model and the one it was compared with. The common failures:
  - generic "okay / no worries, I'll ___" replies that don't answer the message (40–51 of 100 replies vs 9 in the team's);
  - mixing up who does what (e.g. "dabao your food" → "I'll dabao for you");
  - not answering direct questions (where are you, which room).
- Only 190 team-written items in stage 2. The adapter over-fits within about 10 epochs.
- Stage 1 `LOCAL` data (CoSEM) is noisy: in a hand-check of 40 pairs, 13 replies did not answer their message.
- CoSEM and NUS SMS are real messages with automatic scrubbing only, so the model may have seen personal details. About 450 NUS SMS anonymisation tags (`<DECIMAL>` etc.) remained in the stage 0 text.
- NUS SMS language is from the 2010s, and CoSEM is from particular online communities, so the Singlish may be dated or narrow.
- The `LOCAL` style can drift into stereotyped markers ("lah", "sia") or come out plain English.

### Recommendations

Show the replies as suggestions only, and keep a person in the loop.
Do not use the model in sensitive conversations. Do not quote training data without checking it for personal details.

## How to Get Started with the Model

The adapter needs the stage 1 checkpoint `artifacts/mt5_base_after_spans_spk/best` as its base (shipped alongside it) and the input format in `v5_input.json`.
Load it through the ReplyCoach code, which builds the input correctly. It is the default checkpoint, so `V5_CHECKPOINT` is optional:

```bash
pip install -e ".[mt5]"
export V5_CHECKPOINT=artifacts/spk1_ul025_s425/best
```

```python
from replycoach.generate.providers import load_v5

generator = load_v5()
for candidate in generator.generate("eh you dabao lunch for me can?", context=["going canteen now"]):
    print(candidate.style, candidate.candidate_text, round(candidate.score, 3))
```

## Training Details

### Training Data

| Stage | Data | Size |
| --- | --- | --- |
| 0 | CoSEM and NUS SMS Corpus text, for span corruption | 96,295 chunks (CoSEM 89,638, NUS SMS 6,657) |
| 1 | DailyDialog reply pairs, tagged `STANDARD`; a random sample of CoSEM reply pairs, tagged `LOCAL` | 156,052 pairs (DailyDialog 76,052, CoSEM 80,000) |
| 2 | Team-written items (`data/dev/train.jsonl`, split `data-v1`), each with a `STANDARD` and a `LOCAL` gold reply | 190 items → 380 pairs, + 180 context pairs |

No LLM-generated data or labels were used.

### Training Procedure

#### Preprocessing [optional]

- The input is the style tag (`STANDARD:` or `LOCAL:`), the earlier turns marked `Friend:` or `You:`, then the message. Sources are cut to 128 tokens and replies to 48.
- Stage 0 masks 15% of tokens in spans of mean length 3 (mT5 span corruption).
- Stage 2 adds unlikelihood training on 89 "echo" replies (the model copying the input) taken from 34 training items, 3 per item and style.

#### Training Hyperparameters

- **Training regime:** fp32 (full precision). mT5 activations overflow under fp16 and the loss turns to NaN. Optimizer: AdamW in all stages.
- **Stage 0:** 2 epochs, lr 3e-4, warmup 500 steps, batch 32
- **Stage 1:** full fine-tune, 5 epochs, lr 5e-4, warmup 1,000 steps, batch 32, gradient clipping 1.0
- **Stage 2:** LoRA rank 16, alpha 32, dropout 0.1 on attention `q` and `v`; lr 1e-3, batch 16, up to 30 epochs with early stopping (patience 5); unlikelihood weight 0.25; seed 425

#### Speeds, Sizes, Times [optional]

| | Stage 0 | Stage 1 | Stage 2 |
| --- | --- | --- | --- |
| Time (training + validation) | 37 min | 3.0 h | 1.5 min |
| Peak GPU memory | 14.6 GB | 18.1 GB | 8.5 GB |
| Best checkpoint | step 6,020 (the last step) | step 19,508 of 24,385 | epoch 5 of 10 |

- Parameters: 584,170,752 at inference (mT5-base 582,401,280 + LoRA 1,769,472).
- Trainable in stage 2: 1,769,472 (0.30%).
- File sizes: the adapter is 7 MB, and the stage 1 base is 2.3 GB in fp32.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The 50 items in `data/dev/validation.jsonl`. The held-out test split (`data/test/test.jsonl`, 60 items) has not been used.
The validation set was also used to choose the epoch and the run, so the results are optimistic.

#### Factors

Results are reported separately for the `STANDARD` and `LOCAL` styles.

#### Metrics

- Validation loss: token-level cross-entropy of the gold reply, averaged over both styles. Lower is better.
- Blind A/B preference: Role D rated two models' replies with the order hidden, then a sign test was run on the wins.

### Results

| | This model | Same setup without unlikelihood (`spk1_lora_cp`) |
| --- | --- | --- |
| Validation loss (mean of 3 seeds) | 2.486 | 2.482 |
| Seed 425 | 2.494 (`STANDARD` 2.157, `LOCAL` 2.831) | 2.486 (`STANDARD` 2.142, `LOCAL` 2.830) |

The blind A/B against `spk1_lora_cp` covered 81 questions, with 19 identical pairs skipped:
- this model 24, `spk1_lora_cp` 15, tie 15, both bad 27;
- sign test p = 0.20.

#### Summary

Unlikelihood training on echo replies left the validation loss about the same. It was preferred 24–15 in blind rating, though that is not significant (p = 0.20).
The comparison with V4 on the same 50 items has not been done yet.
A later model, `full_ul025_s425`, trained stage 1 on all of CoSEM: validation loss 2.409, and 25–21 against this one in blind rating (p = 0.66).

## Model Examination [optional]

Not done.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA GeForce RTX 3090 (24 GB)
- **Hours used:** about 3.6 GPU-hours for the final three-stage chain. The many experimental runs before it are not counted.
- **Cloud Provider:** None; the university's on-premises SLURM cluster
- **Compute Region:** Singapore
- **Carbon Emitted:** about 0.5 kg CO2e. This is a rough estimate: 350 W GPU board power × 3.6 h ≈ 1.3 kWh, at Singapore's grid factor of about 0.41 kg CO2e/kWh.

## Technical Specifications [optional]

### Model Architecture and Objective

- mT5-base: a 12-layer encoder and 12-layer decoder, d_model 768, with a 250k SentencePiece vocabulary.
- Stage 0 objective: span corruption. Stages 1 and 2: next-reply prediction with cross-entropy, and in stage 2 an unlikelihood term on echo replies.
- Decoding: beam search with 4 beams, no repeated 3-grams, up to 48 new tokens.

### Compute Infrastructure

#### Hardware

One NVIDIA GeForce RTX 3090 per run.

#### Software

- Python 3.13, PyTorch with CUDA, Hugging Face Transformers (≥ 4.44), PEFT 0.21.2, SentencePiece
- Training code: `src/replycoach/generate/spans_v5.py`, `train_v5.py`, `finetune_v5.py`

## Citation [optional]

No paper. If you use the data sources, cite:

- Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). DailyDialog: A manually labelled multi-turn dialogue dataset. *IJCNLP*.
- Gonzales, W. D. W., Hiramoto, M., Leimgruber, J. R. E., & Lim, J. J. (2023). The Corpus of Singapore English Messages (CoSEM).
- Chen, T., & Kan, M.-Y. (2013). Creating a live, public short message service corpus: The NUS SMS corpus. *Language Resources and Evaluation*, 47(2).
- Xue, L., et al. (2021). mT5: A massively multilingual pre-trained text-to-text transformer. *NAACL*.

## Glossary [optional]

- **`STANDARD` / `LOCAL`:** the two reply styles, standard English and Singlish.
- **Echo reply:** a reply that mostly copies the message or the context.
- **Unlikelihood training:** a loss term that pushes down the probability of given bad replies.

## More Information [optional]

See `roles/D_transformer/MODEL_CARD.md` in the repository for the course model card.

## Model Card Authors [optional]

CS425 Group 9, Role D

## Model Card Contact

CS425 Group 9, Role D (through the course team)

### Framework versions

- PEFT 0.21.2
