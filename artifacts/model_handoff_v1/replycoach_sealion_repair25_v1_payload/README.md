# replycoach_sealion_repair25_v1

SEA-LION English→Singlish pipeline: original full-data English adapter plus the latest selected step25 Singlish repair adapter. This is one shared base with two switchable adapters.

This folder includes the base weights, selected adapter weights, tokenizer, compatible model code, exact decoding settings, training provenance, upstream notices and10 example input/output pairs. No model download is required. It contains no full training dataset, project test set, credentials or optimizer states. Weights retain their original stored precision; no merging or quantization was applied.

## Run

Tested runtime: Python3.13.1, Torch2.14.1+cu130, Transformers5.19.0, PEFT0.21.2. Install a PyTorch build appropriate for your machine and the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python infer.py --verify-only
python infer.py --device cuda --verify-examples --output /tmp/sealion_example_replay.json
python infer.py --device cuda --inputs your_inputs.json --output your_replies.json
```

Output files must not already exist. The default inputs are the included10 reused validation examples, not an independent accuracy test. A completed exact replay checks packaging, not reply correctness. The package verifier checks every shipped file against manifest.json before model loading. Source checkpoint paths in export_recipe.json are provenance only; the loader reads bundle-relative files.

Inputs are a JSON list. Each object has item_id, context (oldest-first text list), speakers (matching user/peer/other list), and message (incoming peer text). See examples/inputs.json. Output objects contain english, singlish, standard_anchor and generation status. Input budgets are checked without silent truncation. Beam4/no-repeat-ngram3 decoding is fixed to the demonstrated settings.

## Python API

```python
import json
from infer import ReplyPipeline
model = ReplyPipeline(device="cuda")
rows = json.load(open("examples/inputs.json"))
replies = model.generate_pairs(rows)
```

Use both adapters with the included original learned-position SEA-LION implementation. Keep the English draft as the rewrite anchor. The selected candidate still has context/meaning errors.

CUDA inference is tested by the exporter. CPU FP32 loading is implemented; complete model CPU replay/performance is not verified unless verification.json explicitly records it. SEA-LION CPU loading uses approximately12.7GB of raw FP32 base weights plus runtime memory. Original GPU dtype is BF16 for SEA and FP32 for mT5. The packaging test records the actual GPU and runtime in verification.json. Hardware/library changes may affect numerical replay.

The full training specifications are in training_specs.json. Original training checkpoints are unchanged. The new location uses ordinary files, with no symlinks back to the cluster.
