# Role C: V3 DailyDialog baseline

Date: 21 Sep 2026. This is the full V3 experiment, separate from the one-batch pilot smoke test.

Week 4 smoke proof: 20 pilot items yielded 40 style-labelled pairs; one CPU batch of 4 had loss 5.6428 and saved `artifacts/rnn_v3_smoke.pt`. This only checked that the training path worked.

## Data and setup

- Source: [DailyDialog](https://huggingface.co/datasets/roskoN/dailydialog), Li et al. (2017), CC BY-NC-SA 4.0. The local ZIP files were converted to dialogue JSONL and then to next-turn pairs with `scripts/prepare_dailydialog_pairs.py`.
- Original train and validation dialogue splits were kept separate. Training: 11,118 dialogues / 76,052 pairs. Validation: 1,000 dialogues / 7,069 pairs. No ReplyCoach held-out test items were used.
- Model: V3 word-level LSTM encoder-decoder, embedding size 64, hidden size 128, vocabulary size 5,000, maximum source and target length 40 tokens.
- Seed 425; Adam learning rate 0.001; batch size 32; 5 epochs; CPU. Validation loss determined the saved checkpoint.
- Command: `python -m replycoach.generate.train_v3 --train-pairs data/public/dailydialog_train_pairs.jsonl --validation-pairs data/public/dailydialog_validation_pairs.jsonl --epochs 5 --batch-size 32`

## Results

Loss is token-averaged cross entropy under teacher forcing.

| Epoch | Train loss | Validation loss | Seconds |
| ---: | ---: | ---: | ---: |
| 1 | 5.1406 | 4.6946 | 334.6 |
| 2 | 4.5706 | 4.4406 | 318.4 |
| 3 | 4.3495 | 4.3098 | 338.2 |
| 4 | 4.1999 | 4.2280 | 335.0 |
| 5 | 4.0847 | **4.1704** | 329.9 |

Best checkpoint: local `artifacts/rnn_v3_dailydialog.pt`, epoch 5. The full local report is `artifacts/rnn_v3_dailydialog.json`. Both are generated artifacts and are intentionally ignored by Git.

## Generation observations

The model generates grammatical but often generic or contextually wrong replies. For example, for a question about how long a walk to a station takes, it generated `about half past seven`, which gives a clock time rather than a duration. For a question about a nearby bank, it generated `yes i have a lot of money`. Some outputs contain `<unk>`.

These are DailyDialog validation examples, with input descriptions paraphrased here. DailyDialog provides general dialogue supervision only; this checkpoint has not learned ReplyCoach's STANDARD/LOCAL styles. V4 should use the same training and validation files and settings for comparison.
