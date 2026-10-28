# Qwen3 conversational reranker experiment

28 October 2026. The user approved downloading and testing pretrained
`Qwen/Qwen3-Reranker-0.6B`, then fine-tuning a separate LoRA adapter on the existing
reviewed-pair bundle and comparing reply selection again. This is an experiment;
the demo remains on the explicitly selected reviewed MiniLM checkpoint.

## Fixed design

Use one pinned Qwen revision and one English conversational-suitability
instruction for both pretrained and adapted scoring. Preserve Qwen's official
yes/no prompt format and rank by the yes-minus-no logit. Keep Standard and Local
separate; harmless Standard English is acceptable for the light-local style.
Preserve the full candidate reply and truncate only old context when necessary.

Reuse the validated 190-conversation training bundle plus 20 reviewed alternatives:
1,160 pairs, including 760 potentially noisy algorithmic weak negatives. The 50
validation conversations / 300 pairs stay unchanged; the 60 test items are not
opened. No AI scores, reviewer names, free-text comments or validation proposals
are uploaded as training data. No new labels or abbreviation augmentation.

First evaluate raw Qwen on the existing 400 saved proposals. Then train LoRA
on a private Kaggle T4: rank 8, alpha 16, dropout 0.05, attention query/value
projections, AdamW 2e-5, weight decay 0.01, microbatch 2, accumulation 8, maximum
512 tokens, seed 425, four epochs. Select by existing validation pair loss.
Keep initial weights frozen, record adapter hashes and actual optimizer updates,
and verify save/reload equality. Use a short real-model smoke check before GPU
submission. A failed numerical or integrity check must stop the experiment.

Replay the downloaded adapter with the identical scorer, instruction, saved
proposals and existing 4,000 evaluation-only AI ratings. Report pretrained Qwen,
Qwen+LoRA, reviewed MiniLM and V5-only, paired conversation bootstrap intervals,
runtime and changed choices. The user's greeting is a disclosed development
diagnostic, never an automatically added training example. Neither that single
example nor reused validation establishes held-out quality or an automatic
promotion decision. Keep all earlier checkpoints and results.

## Sources

- [Qwen model and official yes/no template](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).
- [Official pointwise reranker training](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md).
- [PEFT LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora).

## Pretrained results

Downloaded revision `e61197ed45024b0ed8a2d74b80b4d909f1255473` (about 1.19 GB
of safetensors). Completed the matched replay on 400 saved proposals from 50
validation conversations, using the same 4,000 existing AI-persona ratings and
zero new judge calls. No test items or AI-derived training labels were used.

| Selector | Standard appropriateness /5 | Local appropriateness /5 | Standard would-send | Local would-send |
| --- | ---: | ---: | ---: | ---: |
| Reviewed MiniLM | 3.508 | 3.930 | 55.4% | 71.0% |
| Pretrained Qwen, conversational instruction | 3.578 | 4.044 | 58.8% | 72.4% |
| Fixed V5 | 3.666 | 4.110 | 61.2% | 76.8% |

Pretrained Qwen minus reviewed MiniLM appropriateness differences, descriptive
95% paired conversation-bootstrap intervals (20,000 resamples, seed 425):
Standard **+0.070 [-0.122, 0.288]**; Local **+0.114 [-0.192, 0.456]**. Point
estimates improve but both intervals include zero. V5-only retains higher means.
Repeated validation, correlated AI-persona ratings and a changed architecture
plus instruction limit inference. This is not a held-out or human evaluation.

Observed CPU replay time: 442.79 seconds, plus 29.35 seconds of loading. Other
software checks ran concurrently, so this is not a controlled latency benchmark.

On the separately disclosed greeting diagnostic (not inserted into training),
raw Qwen scores “I'm fine, thanks” at **3.115**, versus “I 到了, you where ah?”
at **-5.704** for the requested Local style. This fixes the observed MiniLM
ordering on that example, not proof of general conversational competence.

## Software and private training status

The full suite passed 415 tests before submission. A real, pinned Qwen CPU
smoke run made one adapter update, preserved frozen parameters, saved a checkpoint
explicitly marked nonproduction, and reloaded identical adapter parameters.
Smoke losses and its adapter are not production-quality evidence.
After the runtime setup correction, all 29 focused Qwen, shared-ranking and
selection-replay tests passed again (13.39 seconds).

- [Private minimal dataset](https://www.kaggle.com/datasets/tanjingminjereme/replycoach-qwen-pairs-20261028-01), version 1.
- [Private T4 notebook](https://www.kaggle.com/code/tanjingminjereme/replycoach-qwen-lora-20261028-01).
- Dataset and notebook privacy, GPU enablement and T4 assignment were read back
  using the official authenticated Kaggle MCP. The upload hashes and merged
  training-pair hash match the reviewed MiniLM experiment exactly.
- Notebook version 1 failed before any updates because Kaggle's optional
  `torchao==0.10.0` conflicts with PEFT, which requires newer torchao when present.
  Version 2 removes this unused optional package from the temporary GPU runtime;
  no quantization is used. Training settings, data and trainer files are unchanged.
  The notebook packaging regression test passes with this setup correction.
- Version 2 completed successfully on Tesla T4. The downloaded adapter and base
  file hashes, identical GPU training rows, and actual reloaded adapter tensor
  hash were verified. The adapted reply-selection replay also completed; the
  demo has not been changed.

## Completed LoRA training

All four epochs completed, with 292 actual optimizer updates. The best weak-pair
validation loss was at epoch 4. Training/validation/checkpoint saving took 723.17
seconds after loading/setup. This timing excludes notebook startup, installation
and base-model download. Only 1,146,880 LoRA parameters trained; frozen base
parameters remained unchanged. The adapter file is about 4.39 MiB.

| Epoch | Training pair loss | Validation pair loss | Total updates |
| --- | ---: | ---: | ---: |
| 1 | 0.42877 | 0.38809 | 73 |
| 2 | 0.36650 | 0.36874 | 146 |
| 3 | 0.33660 | 0.36334 | 219 |
| 4 | 0.30227 | **0.34906** | 292 |

The adapter is saved separately at
`artifacts/reranker_qwen/kaggle_output_20261028_01_v2/qwen_lora/best/`.
Its safetensors SHA-256 is
`b05450649150cb0918f5769d5176f0dc71973600f2fa0803d35b76d34e7f7207`.
The unchanged base-file SHA-256 is
`27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b`.
Downloaded training rows are byte-identical to the approved 1,160-row local
training payload. The source/prompt and development fingerprints also match.

The adapted checkpoint still orders the greeting correctly: “I'm fine, thanks”
scores 2.654 versus -2.850 for the unrelated arrival reply. These logits are
uncalibrated ranking signals; their scales should not be compared across models
as confidence. This diagnostic was never added to training.

## Completed matched adapted comparison

The downloaded epoch-4 adapter replayed the identical 400 saved proposals / 50
validation conversations. The scorer verified the actual adapter tensor hash
as well as its serialized file and matching base file. Prompt/data fingerprints,
packet/source-export hashes and all 4,000 existing rating records match the raw
replay. No new judge calls or test-set examples were used. Selected contexts
were not truncated in either replay.

| Selector | Standard appropriateness /5 | Local appropriateness /5 | Standard would-send | Local would-send |
| --- | ---: | ---: | ---: | ---: |
| Reviewed MiniLM | 3.508 | 3.930 | 55.4% | 71.0% |
| Pretrained Qwen | 3.578 | **4.044** | 58.8% | **72.4%** |
| Qwen + LoRA | **3.640** | 3.946 | **60.8%** | 70.0% |
| Fixed V5 | 3.666 | 4.110 | 61.2% | 76.8% |

Paired differences, with the same descriptive 95% conversation-bootstrap
intervals (20,000 resamples, seed 425):

| Comparison | Standard appropriateness | Local appropriateness |
| --- | --- | --- |
| LoRA minus raw Qwen | +0.062 [-0.006, 0.148] | -0.098 [-0.360, 0.108] |
| LoRA minus reviewed MiniLM | +0.132 [-0.064, 0.372] | +0.016 [-0.232, 0.286] |
| LoRA minus V5-only | -0.026 [-0.412, 0.372] | -0.164 [-0.466, 0.140] |

LoRA changes four Standard and five Local selections. It improves the Standard
point mean but reduces the Local mean compared with the pretrained Qwen model.
All these appropriateness intervals include zero. **An overall improvement from
LoRA is not established.** V5-only still has the highest appropriateness point
means. Keep raw Qwen, the adapter and earlier MiniLM checkpoints separately;
no automatic model promotion or further tuning was performed.

Examples from the nine changed decisions:

- ITEM-B-056 Local: after the peer says the badminton court is booked for seven,
  LoRA selects “okay, seven works for me” instead of the unrelated waiting reply.
  Existing AI means change 4.2 to 5.0.
- ITEM-B-034 Local: after You promise an update about a missed class, the peer
  asks “can tell me what I miss?” LoRA selects “you can send me the details”,
  reversing the expected speaker responsibility. Existing means change 4.9 to
  1.0. The raw selection mentioning a link is also not an ideal reference answer.

All changes, including regressions and questionable AI ratings, are preserved in
`lora_selection_changes_20261028.json`. Neither the isolated greeting success nor
lower pair loss establishes robust conversational reasoning. The dataset still
contains 760 weak negatives, only seven new human-confirmed negatives, one seed,
and repeated use of the same validation conversations. The AI-persona ratings
are correlated and sometimes implausible; they are diagnostic development
evidence, not independent human preferences. This experiment also cannot
separate architecture, size, pretraining and instruction effects.

Training pairs contain human-written reference/donor replies, while deployment
also ranks generated replies and safety fallbacks. Binary suitability labels
do not explicitly order multiple suitable alternatives. These supervision and
candidate-distribution differences are plausible limitations; this experiment
does not isolate them as causes or show that additional epochs would fix them.

Observed adapted CPU replay time was 480.95 seconds, with 40.23 seconds of
initial loading and the disclosed greeting/integrity checks. Hardware contention
was not controlled between runs; these are observed run times, not a rigorous
latency comparison. Both replays used the same CPU scorer precision and token
budget; GPU training used the separate temporary T4 environment.

**Status:** download, pretrained testing, private GPU LoRA training, adapter
verification and matched adapted testing are complete. The requested comparison
is documented for the final report. The demo remains on reviewed MiniLM.

## Local evidence and reproduction

Ignored evidence is under `artifacts/reranker_qwen/`: pinned source/model files,
`replay_raw_20261028_01/result.json`, `raw_greeting_20261028.json`,
`comparison_20261028.json`, `replay_lora_20261028_01/result.json`,
`lora_selection_changes_20261028.json`, `lora_greeting_20261028.json`,
`real_model_smoke_20261028_01/`, the allowlisted Kaggle
package and privacy readback, notebook failure log, and full test output.

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Reranker-0.6B', revision='e61197ed45024b0ed8a2d74b80b4d909f1255473', local_dir='artifacts/reranker_qwen/pretrained', allow_patterns=['*.safetensors', 'config.json', 'generation_config.json', 'tokenizer*.json', 'vocab.json', 'merges.txt'])"
.\.venv\Scripts\python.exe scripts/replay_shared_ranker.py --packet experiments/judging/four_systems_20261026_01/main_packet --qwen-model artifacts/reranker_qwen/pretrained --out-dir artifacts/reranker_qwen/new_raw_replay
.\.venv\Scripts\python.exe scripts/package_qwen_reranker.py --bundle artifacts/reranker_v6/kaggle_20261027_reviewed01/dataset --output artifacts/reranker_qwen/new_package --owner YOUR_KAGGLE_SLUG --tag YYYYMMDD-NN
```

Use fresh output directories. The fine-tuned replay additionally takes
`--qwen-adapter PATH_TO_DOWNLOADED_BEST`; the loader verifies base/adapter hashes,
prompt identity and completed production training provenance. The notebook uses
PEFT 0.21.2 and Transformers 5.19.0; the existing demo does not load Qwen.
