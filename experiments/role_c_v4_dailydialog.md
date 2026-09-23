# Role C: V4 attention-RNN on DailyDialog

Date: 23 Sep 2026. This experiment adds Luong dot-product attention to the V3 encoder-decoder and masks padded source positions before the attention softmax.

## Data and setup

- Source: DailyDialog train and validation splits prepared as next-turn response pairs.
- Training: 11,118 dialogues / 76,052 pairs. Validation: 1,000 dialogues / 7,069 pairs. No ReplyCoach held-out test items were used.
- Model: word-level LSTM encoder-decoder with attention, embedding size 64, hidden size 128, vocabulary size 5,000, maximum source and target length 40 tokens, 1,516,552 parameters.
- Seed 425; Adam learning rate 0.001; batch size 32; 5 epochs; CUDA. Validation loss determined the saved checkpoint.
- Command: `python -m replycoach.generate.train_v3 --train-pairs data/public/dailydialog_train_pairs.jsonl --validation-pairs data/public/dailydialog_validation_pairs.jsonl --attention --epochs 5 --batch-size 32`

## Results

Loss is token-averaged cross entropy under teacher forcing.

| Epoch | Train loss | Validation loss | Seconds |
| ---: | ---: | ---: | ---: |
| 1 | 5.1181 | 4.6000 | 174.2 |
| 2 | 4.4655 | 4.3264 | 173.9 |
| 3 | 4.2274 | 4.1902 | 174.2 |
| 4 | 4.0676 | 4.1125 | 172.6 |
| 5 | 3.9401 | **4.0573** | 172.2 |

Best checkpoint: `rnn_v4_dailydialog.pt`, epoch 5. The checkpoint and full generated JSON report remain outside Git under ignored `artifacts/` storage.

## Generation examples

| Reference | V4 generated reply |
| --- | --- |
| `There is one . 5 blocks away from here ?` | `yes please` |
| `Well , that's too far.Can you change some money for me ?` | `yes it is` |
| `Surely , of course . What kind of currency have you got ?` | `sure` |
| `RIB .` | `i have a <unk>` |
| `How much would you like to change ?` | `i would like to have a credit card` |

The outputs are generally fluent but remain short, generic, or contextually incorrect. Attention lowered validation loss, but it did not resolve the word-level vocabulary limitation or guarantee an appropriate response. DailyDialog supplies generic conversation supervision and does not teach ReplyCoach's STANDARD/LOCAL styles.
