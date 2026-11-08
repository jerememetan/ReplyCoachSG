# New transformer bundles in the demo

The November 8 transformer handoff is available as two separate generator choices:
**mT5 A (new handoff)** and **SEA-LION 3B (new handoff)**. In Streamlit, select
**Generator only** or **Hybrid**, then choose the model. The original V5 option,
All models comparison, V6/V7 pipeline and frozen evaluation entry points retain
their existing checkpoints and ranking policy. These new bundles have not been added
to the historical evaluation results.

## Local files

The full private bundles belong at:

- `artifacts/model_handoff_v1/replycoach_mt5_A_v1/`
- `artifacts/model_handoff_v1/replycoach_sealion_repair25_v1/`

Each directory must contain `manifest.json`, `infer.py`, base weights, tokenizer,
adapter(s), and its bundled source. Both directories are excluded from Git.
Optional path overrides are `REPLYCOACH_MT5_A_BUNDLE` and
`REPLYCOACH_SEALION_BUNDLE`.

The provider runs the exact bundle loader in a persistent isolated Python process
using the same Python environment as Streamlit. This prevents its bundled
`replycoach` modules from replacing the demo's modules. The first request checks
the manifest hashes and loads the model. Later requests reuse the loaded model.
Logs are saved under `output/handoff_logs/`. Restart Streamlit to release loaded
workers or pick up changed bundle files.

Use a compatible local environment; see the pinned requirements supplied with
each bundle. Do not install the cluster's CUDA Torch build over a local CPU build
without checking machine compatibility. The bundle chooses CUDA if available,
otherwise CPU. This does not connect the demo to the school GPU automatically.

## Output and limits

The new sources are identified as `MT5_A` and `SEALION` in candidate metadata.
Each loader produces a Standard reply and a Singlish rewrite anchored to that
Standard reply. Failed, empty or length-limited stages are reported as generator
errors through the existing system error path; no replacement model is loaded.
The bundle does not supply calibrated candidate probabilities: candidates use
the neutral score `0.5`, explicitly marked `unscored_bundle_output`, before the
existing system ranking policy. This score is not a model quality rating.

The demo supplies plain text history, so speaker roles are inferred by alternating
back from the incoming peer message, matching the existing V5 convention. The
bundle's decoding, prompt format, adapter switching and token limits are retained.
Local CPU speed and output quality should be assessed separately from transfer
checksum verification and the teammate's GPU example replay.

During the initial local smoke check, mT5 produced a complete Standard/Local pair
in 25.17 seconds on CPU, including process startup, checksum verification and
model loading. This is one example, not a benchmark. The current machine has
approximately 16 GB RAM; SEA-LION's FP32 base weights alone need about 12.7 GB,
so its full local CPU inference was not used as a validation step.

## Verified laptop GPU setup (November 8)

The project's `.venv` now uses `torch==2.14.1+cu130` on the NVIDIA GeForce RTX
5050 Laptop GPU (8 GB), driver 610.71. A BF16 CUDA matrix multiplication and both
real handoff models passed local GPU smoke checks. To reproduce the installation:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m streamlit run src/replycoach/demo_app.py --server.fileWatcherType none
```

Restart any already-running Streamlit process after changing PyTorch. The demo
automatically chooses CUDA when available, including its V4 provider and shared
MiniLM/Qwen rerankers. The default scorer constructors used outside the demo
continue to default to CPU. No checkpoint weights, prompts or grading criteria
were changed. GPU arithmetic may produce small numerical differences; the
previous frozen evaluation files have not been rerun or overwritten.
Both final-evaluation entry points explicitly retain the frozen CPU reranker;
the V7 extension checks its identity against the saved V6 selection before use.

Switching between the baseline models, mT5 A and SEA-LION closes old bundle
workers, clears cached model resources and releases unused GPU memory before
loading the selected group. This avoids keeping both large handoff models on
the 8 GB card. Returning to a previous model requires loading it again.

| Single-example smoke check | First request, including verification/loading | Reused model request | Peak model GPU allocation |
|---|---|---|---|
| mT5 A | 17.08 s | 0.69 s | 2.41 GB |
| SEA-LION 3B | 29.58 s | 2.42 s | 6.84 GB |

These are two requests for one example per model, not a benchmark or new quality
evaluation. Peak allocation excludes CUDA driver/display overhead. Evidence is
saved locally at `output/gpu_handoff_smoke.json`. Neither model requires the
school cluster for this local setup.
