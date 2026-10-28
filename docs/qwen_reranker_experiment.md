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
- Version 2 has been submitted. Completed GPU training and adapted replay results
  are still pending; the demo has not been changed.

## Local evidence and reproduction

Ignored evidence is under `artifacts/reranker_qwen/`: pinned source/model files,
`replay_raw_20261028_01/result.json`, `raw_greeting_20261028.json`,
`comparison_20261028.json`, `real_model_smoke_20261028_01/`, the allowlisted Kaggle
package and privacy readback, notebook failure log, and full test output.

```powershell
.\.venv\Scripts\python.exe scripts/replay_shared_ranker.py --packet experiments/judging/four_systems_20261026_01/main_packet --qwen-model artifacts/reranker_qwen/pretrained --out-dir artifacts/reranker_qwen/new_raw_replay
.\.venv\Scripts\python.exe scripts/package_qwen_reranker.py --bundle artifacts/reranker_v6/kaggle_20261027_reviewed01/dataset --output artifacts/reranker_qwen/new_package --owner YOUR_KAGGLE_SLUG --tag YYYYMMDD-NN
```

Use fresh output directories. The fine-tuned replay additionally takes
`--qwen-adapter PATH_TO_DOWNLOADED_BEST`; the loader verifies base/adapter hashes,
prompt identity and completed production training provenance. The notebook uses
PEFT 0.21.2 and Transformers 5.19.0; the existing demo does not load Qwen.
