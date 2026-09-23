# Role C: V3 versus V4

Date: 23 Sep 2026.

## Controlled setup

V3 and V4 used the same DailyDialog training and validation files, vocabulary limit, token limits, model dimensions, seed, optimizer, learning rate, batch size, and five-epoch schedule. The architectural change was the addition of Luong dot-product attention with a source padding mask in V4.

V3 ran on CPU and V4 ran on CUDA. Their elapsed times therefore do not measure the speed effect of attention and are excluded from the model decision.

## Validation comparison

| Epoch | V3 validation loss | V4 validation loss | V4 minus V3 |
| ---: | ---: | ---: | ---: |
| 1 | 4.6946 | 4.6000 | -0.0946 |
| 2 | 4.4406 | 4.3264 | -0.1142 |
| 3 | 4.3098 | 4.1902 | -0.1196 |
| 4 | 4.2280 | 4.1125 | -0.1155 |
| 5 | 4.1704 | **4.0573** | **-0.1131** |

V4 had lower validation loss at every epoch. Its best loss was 0.1131 lower than V3, a 2.71% relative reduction from the V3 value. Both models reached their best validation loss at epoch 5.

## Generation comparison

The following outputs use the same first five validation examples.

| Reference | V3 generated | V4 generated |
| --- | --- | --- |
| `There is one . 5 blocks away from here ?` | `yes i have a lot of money` | `yes please` |
| `Well , that's too far.Can you change some money for me ?` | `yes i have a lot of <unk>` | `yes it is` |
| `Surely , of course . What kind of currency have you got ?` | `sure i have a <unk>` | `sure` |
| `RIB .` | `i want to go to the <unk> of the <unk>` | `i have a <unk>` |
| `How much would you like to change ?` | `i want to have a <unk>` | `i would like to have a credit card` |

V4 often produced shorter and more fluent text and used fewer unknown tokens in this small sample. The replies were still frequently generic or semantically incorrect. The qualitative sample does not show a decisive improvement in contextual appropriateness.

## Decision and limitations

V4 is selected over V3 as the stronger RNN generator because it improved validation loss consistently under the matched data and hyperparameter setup. This is a provisional RNN-branch decision: V4 still needs comparison with V5 and task-specific evaluation before the final system generator is selected.

Limitations include greedy decoding, a 5,000-word vocabulary, only five displayed examples from one validation dialogue, and pretraining on generic DailyDialog rather than STANDARD/LOCAL ReplyCoach data. Neither V3 nor V4 has been evaluated on the held-out ReplyCoach test set.
