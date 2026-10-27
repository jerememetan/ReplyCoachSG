# V6 MiniLM cross-encoder: training and validation handoff

27 October 2026. **The proposal's V6 fine-tuning experiment is now trained.**
The separate Fine-tuned V6 demo option is experimental: its validation means
improve over the frozen reranker, while V5-only remains higher on average.

An additional, separate [abbreviation-augmentation experiment](../../docs/reranker_augmentation.md)
uses the existing short-form mapping to expand training inputs. Its notebook,
settings, actual expansion counts and results are documented separately so the
original unaugmented checkpoint and measurements below remain identifiable.

## Kaggle run and checkpoint

- Private notebook: [ReplyCoach V6 MiniLM fine-tuning](https://www.kaggle.com/code/tanjingminjereme/replycoach-v6-minilm-fine-tuning-20261027), version 1, COMPLETE.
- Private dataset: [V6 training bundle](https://www.kaggle.com/datasets/tanjingminjereme/replycoach-v6-training-20261027), version 1. Both privacy settings were read back through the authenticated official MCP.
- Device: Tesla T4; PyTorch 2.11.0+cu128, Transformers 5.19.0, Tokenizers 0.23.2.
- Initialization: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- One encoder and scoring head for both styles; actual memory speakers and the
  requested style are included in the input. No generator was retrained.
- AdamW, learning rate 2e-5, weight decay 0.01, batch16, maximum pair length512,
  seed425, four epochs, 288 optimizer steps. Preserve the reply on truncation.
- Best checkpoint: **epoch 2**, selected solely by validation pair loss. The
  epoch loop, validation and checkpoint saving took 15.21 seconds after loading;
  package installation, model download and notebook startup are additional time.

| Epoch | Training pair loss | Validation pair loss | Total updates |
| --- | ---: | ---: | ---: |
| 1 | 0.71956 | 0.59801 | 72 |
| 2 | 0.45494 | **0.48777** | 144 |
| 3 | 0.38348 | 0.51287 | 216 |
| 4 | 0.30465 | 0.59567 | 288 |

The divergence after epoch2 is consistent with overfitting, motivating selection
of epoch2. These pair losses are not reply-quality or accuracy percentages.
The final run completed all four epochs and retained the best saved checkpoint.

The installed private checkpoint is `artifacts/reranker_v6/best/`; override it
with `REPLYCOACH_V6_CHECKPOINT`. It contains safetensors weights, tokenizer/config
and `v6_training.json`. The loader requires a completed GPU-training manifest,
real optimizer updates, changed weights and a matching tensor-weight hash. It
refuses CPU smoke checkpoints. The preceding local check used the actual pinned
model for two optimizer steps and was explicitly marked nonproduction.

## Supervision and data boundaries

The 190 reviewed training conversations yield 380 positive context/reply pairs
and 760 deterministic mismatches: 1,140 training pairs total. The 50 validation
conversations yield 100 positives and 200 mismatches: 300 validation pairs.
Every negative donor comes from the training pool and uses the same requested
style; both known positive texts and duplicate normalized donor texts are skipped.
The user explicitly approved this **algorithmic weak supervision**. Mismatches
are not independently human-verified negatives and may include valid replies.
The extra pairs do not create extra independent conversations.

The minimal private upload excludes author/reviewer names, review comments,
credentials, model weights and all judge ratings. The 60 test items were not
opened. All existing 4,000 Gemma ratings remain evaluation-only and were not
used for training or selecting the best epoch. The original eight source/data/
rubric hashes and ratings hash were checked unchanged after the experiment.

## Saved-proposal reply-quality comparison

After training, replayed the exact 400 already-judged proposals on the same 50
validation conversations. One Standard/Local pair is selected per conversation.
The original 4,000 AI ratings are joined only after selection: **zero new judge
calls**. This is validation analysis on fixed saved source outputs, not a fresh
end-to-end generation experiment or held-out test result.

| Selector | Standard appropriateness /5 | Local appropriateness /5 | Standard would-send | Local would-send |
| --- | ---: | ---: | ---: | ---: |
| Frozen MiniLM, original input | 3.060 | 3.796 | 43.8% | 67.4% |
| Frozen MiniLM, style-aware input | 2.970 | 3.810 | 40.2% | 67.4% |
| **Fine-tuned V6, style-aware input** | **3.528** | **3.878** | **55.4%** | **68.6%** |
| Fixed V5 | 3.666 | 4.110 | 61.2% | 76.8% |
| Fixed MiniLM retrieval | 2.976 | 3.802 | 40.0% | 65.0% |
| Legacy mixed-score selector | 2.678 | 3.476 | 29.0% | 53.8% |

The frozen style-aware control separates the input-format change from weight
training. Paired appropriateness differences, with descriptive 95% percentile
conversation-bootstrap intervals (20,000 resamples, seed425):

| Fine-tuned minus comparator | Standard | Local |
| --- | --- | --- |
| Original frozen scorer | +0.468 [0.106, 0.860] | +0.082 [-0.080, 0.252] |
| Style-aware frozen scorer | +0.558 [0.182, 0.956] | +0.068 [-0.098, 0.240] |
| Fixed V5 | -0.138 [-0.514, 0.238] | -0.232 [-0.608, 0.126] |

The Standard improvement over frozen scoring is supported by these conditional
intervals; the Local gain is unresolved. V5-only has higher point estimates,
and its appropriateness differences from V6 are also unresolved by these
intervals. **The hypothesis that hybrid selection beats the best generator is
not established.** No tuning was performed to force that conclusion.

Ten personas share one AI judge and are correlated. Bootstrap units are the 50
conversations, not 4,000 independent humans. The validation set also selected
the checkpoint using weak pair loss, so these are development results. Rates
are simulated willingness to send, not human approval. Weak negatives, small
data, earlier generator speaker-guessing and possible factual inventions remain
limitations. The trained model is not a calibrated confidence estimator.

## Use and reproduction

Launch the existing demo and choose **System settings → All models
(experimental) → Shared selector → Fine-tuned V6**. Select the reviewed bank
`data/dev/train.jsonl`. The default remains Frozen MiniLM in All models, and the
overall default remains Retrieval comparison. Missing checkpoint errors are
visible and do not silently substitute the frozen scorer.

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/replycoach/demo_app.py --server.fileWatcherType none
.\.venv\Scripts\python.exe scripts/replay_shared_ranker.py --packet experiments/judging/four_systems_20261026_01/main_packet --checkpoint artifacts/reranker_v6/best --out-dir artifacts/reranker_v6/new_finetuned_replay
.\.venv\Scripts\python.exe scripts/replay_shared_ranker.py --packet experiments/judging/four_systems_20261026_01/main_packet --style-aware-baseline --out-dir artifacts/reranker_v6/new_frozen_style_replay
```

Every replay output directory must be fresh. Checkpoint data fingerprints must
match the current reviewed train/validation files. The original frozen replay
command remains unchanged. For a new training run, use `prepare_reranker_training.py`
then `package_reranker_kaggle.py`; their `--help` documents required paths/owner/
unique run tag. The checked-in notebook has no saved outputs. Launch commands
used for this run, from the corresponding generated directories:

```powershell
uv tool run --from kaggle kaggle datasets create -p . --keep-tabular
uv tool run --from kaggle kaggle kernels push -p . --accelerator NvidiaTeslaT4 --timeout 1800
```

On this Windows CLI version, a nested relative `-p` path caused upload-manifest
errors. Running from the generated dataset/notebook folder with `-p .` succeeded.
No `--public` option was used; notebook metadata explicitly sets private=true.

Private evidence under `artifacts/reranker_v6/` includes the exact uploaded bundle,
package code hashes, privacy readbacks, GPU logs/output, local smoke result,
checkpoint, both new replay results and `selector_comparison_20261027.json`.
Original training code is preserved in private Kaggle dataset version1.

Kaggle MCP is registered locally at the official endpoint. Codex's default OAuth
client ID was rejected; the user's successful official CLI login is now reused
through a local `http_headers_helper`. An authenticated private-resource MCP
request succeeded. Credentials/helper configuration live outside this repo;
new Codex sessions can load that configuration. No credentials are embedded in
the notebook or this handoff.

Software verification: **370 tests passed in 227.55 seconds**, including bundle
guards, provenance, real optimizer updates, non-finite loss rejection, private
packaging, smoke-checkpoint rejection, style-aware scoring and demo selector
switching. Checkpoint loading and both real model replays also succeeded.

The live Streamlit check also succeeded with the 190-item reviewed bank: all
four source models returned their two proposals, the displayed scorer identity
matched the installed trained-weight hash, and clicking a selected reply filled
the preview composer without changing the conversation. For the supplied lunch/
library example, V6 chose V5's Standard reply and the shared BM25/MiniLM Local
reply. V4's Local proposal was visibly marked as a safety fallback. That single
example verifies integration, not overall reply quality. Screenshots are saved
locally under ignored `docs/superpowers/screenshots/2026-10-27-finetuned-v6*.png`.
