# Receiver-side V5 verification - 26 October 2026

## Confirmed locally

The existing `load_v5()` provider on main loaded the supplied mT5-base checkpoint
and `spk1_ul025_s425` LoRA adapter. This was real inference, not reading cached
reply strings as model output. No model weights, configuration, authored data,
saved ratings or original exports were modified. No training or hosted inference
calls were made. Hugging Face/Transformers were forced offline for the check.

| Component | Installed local file |
| --- | --- |
| Adapted base | `artifacts/mt5_base_after_spans_spk/best/model.safetensors` |
| LoRA adapter | `artifacts/spk1_ul025_s425/best/adapter_model.safetensors` |
| Tokenizer/configuration | Matching files in both `best/` folders |
| Input-format flags | `v5_input.json`: speaker labels enabled, joint output disabled |

The loaded model contained 584,170,752 parameters, matching the model card.
It ran on CPU: the installed `torch 2.14.0+cpu` build cannot use CUDA even though
the laptop has an NVIDIA GPU. CPU threads were limited to two for this check.

## Saved-output reproduction

The existing generator was called with the original message and selected memory
from the teammate's export, preserving its existing speaker-format behaviour.

| Validation item | Standard text | Local text | Generation time, both styles |
| --- | --- | --- | --- |
| ITEM-A-017 | Exact match | Exact match | 4.51 s |
| ITEM-A-038 | Exact match | Exact match | 6.44 s |
| ITEM-C-038 | Exact match | Exact match | 6.00 s |

All six decoded reply texts matched. Input-score absolute differences from the
saved export were between 1.42e-5 and 3.10e-5; scores are not claimed bitwise
identical across CPU and the sender's environment. Initial provider loading took
50.92 seconds, including its imports and weight loading. These three-item timings
exclude the other retrievers/ranker and are not an end-to-end demo benchmark.

## Dependencies and evidence

Verified versions: torch 2.14.0, transformers 5.19.0, peft 0.21.2,
sentencepiece 0.2.2 and accelerate 1.15.0. Missing PEFT/SentencePiece dependencies
were installed into the existing project virtual environment; no CUDA runtime
or GPU PyTorch replacement was installed.

Evidence is saved privately in the existing ignored evaluation folder:

- `experiments/judging/four_systems_20261026_01/v5_local_verification_20261026.json`
- `experiments/judging/four_systems_20261026_01/v5_local_verification_console.txt`

The evidence contains exact outputs, saved outputs, model scores, text-match
checks, package versions, timings and original generator metadata.

The loader warned that shared embeddings and the language-model head contain
different values despite the saved tie-word-embeddings configuration. Transformers
retained the distinct supplied weights, and all six outputs matched. The
checkpoint configuration was deliberately left unchanged. Confirm this detail
with Role D before any later checkpoint conversion or re-saving.

## Limits

This establishes that the received provider and weights can run locally and
reproduce a fixed three-conversation subset. It does not independently reproduce
the training history, all 100 validation outputs, every device environment or
future input quality. V5's plain-string API still guesses alternating speakers;
consecutive same-speaker turns remain a known input-format limitation. Model
quality comparisons are in `docs/reports/four_system_comparison_20261026.md`.
