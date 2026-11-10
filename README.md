# ReplyCoachSG

ReplyCoachSG is an SMU CS425 project for casual peer coordination. Give it a
recent conversation and an incoming message to get a **Standard English**
reply and a **Light Local** reply with Singlish wording. Selecting a suggestion
puts it in the demo's chat composer for preview.

## Run the demo

Use Python 3.12 and open the project folder in VS Code. Run these commands from
the folder containing `pyproject.toml`.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[demo]"
.\.venv\Scripts\python.exe -m streamlit run src/replycoach/demo_app.py --server.fileWatcherType none
```

**macOS or Linux**

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[demo]"
./.venv/bin/python -m streamlit run src/replycoach/demo_app.py --server.fileWatcherType none
```

Open [http://localhost:8501](http://localhost:8501). The default **Retrieval
comparison** runs BM25 and MiniLM against the **Reviewed development file**
(`data/dev/train.jsonl`, 190 examples). This mode works from the repository
without the private checkpoint handoff. MiniLM may need internet access to
download its public embedding model on first use. For BM25 only, select
**Retrieval only** and choose **BM25**.

## Run the full presentation demo

The full comparison uses weights distributed separately by the project team.
Copy or extract the **complete** checkpoints into these locations:

| Checkpoint | Location |
| --- | --- |
| V4 attention RNN | `artifacts/rnn_v4_task.pt` |
| SEA-LION 3B inference bundle | `artifacts/model_handoff_v1/replycoach_sealion_repair25_v1/` |
| Qwen3 reranker base and tokenizer | `artifacts/reranker_qwen/pretrained/` |
| Qwen DailyDialog adapter and training manifest | `artifacts/reranker_qwen/dd_result_20261029/stage2_reviewed/best/` |

The SEA-LION folder should contain `infer.py` and `manifest.json` directly, next
to the bundle's base model, tokenizer, adapters and source. See the
[checkpoint handoff guide](artifacts/model_handoff_v1/README.md) for archive
verification and the [GPU setup guide](docs/transformer_handoff_demo.md) for the
tested CUDA environment. Use a CUDA-enabled PyTorch build for a compatible
NVIDIA GPU. SEA-LION's FP32 weights alone require about 13 GB of RAM on CPU.

Install the additional inference dependencies in the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[demo,mt5,qwen]" "einops==0.8.2"
```

On macOS or Linux, substitute `./.venv/bin/python`. Restart Streamlit after
installing dependencies or replacing a checkpoint. The first request loads
the models, so allow extra time for that request. Gemma and Ollama are used
for offline grading and are optional for running the demo.

1. Under **System settings**, choose **All models (experimental)**.
2. Keep **Memory** on **M2** and **Reply bank** on **Reviewed development file**.
3. Enter the **Recent conversation** and **Incoming message**. Use one turn per
   line in the conversation, prefixed with `You:` or `Peer:`.
4. Click **Suggest two replies**. Open **Local wording candidates** to see
   which variants were scored. Click a reply to preview it in the composer.

Try this example:

**Recent conversation**

```text
You: hi can I lend some of your equipments please?
Peer: can ah, but i need it back by next week
You: can will return it by then
```

**Incoming message**

```text
please be reminded that you have to return the equipment by tomorrow 9pm. If not you gg liao.
```

**Standard** comes directly from SEA-LION 3B. **Light Local** starts from its
Local reply. V7 v2 tries dictionary substitutions such as `eat` → `makan` or
`tomorrow` → `tmr`. If it produces three distinct edits, Qwen ranks them
alongside the original. Otherwise the app keeps the original. The comparison
table also shows BM25, MiniLM and V4 proposals. If SEA-LION fails to load, the
table can still show available proposals and the app reports the model error.
[V7 v2 method and fallback behavior](docs/v7_random_lexical.md)

## Results

| Final demo output | Appropriateness /5 | Naturalness /5 | Style fit /5 | Would send |
| --- | ---: | ---: | ---: | ---: |
| Standard · V5 SEA-LION | 4.555 | 4.975 | 5.000 | 88.0% |
| Light Local · V7 v2 | 4.603 | 4.908 | 3.947 | 81.2% |

These are simulated ratings from ten Gemma personas on 60 previously inspected
test conversations. The Local follow-up used one reproducible wording sample
per conversation. [Full V7 v2 evaluation and comparison table](docs/evaluation/v7_v2_local_20261109.md)

- [Transformer selection on 50 validation conversations](docs/evaluation/transformer_selection_20261108.md)
- [Six-system evaluation including historical V7 v1](docs/evaluation/current_pipeline_20261108.md)
- [Architecture and version history](docs/architecture.md)
- [Evaluation entry points and data splits](docs/final_test_evaluation.md)

## Repository map

| Path | Contents |
| --- | --- |
| [`src/replycoach/`](src/replycoach/) | Demo, retrieval, generation and ranking code |
| [`data/`](data/) | Reply bank, evaluation splits, lexicons and rubric |
| [`docs/evaluation/`](docs/evaluation/) | Consolidated findings and aggregate results |
| [`experiments/`](experiments/) | Research runs and supporting records |
| [`scripts/`](scripts/), [`tests/`](tests/) | Training, evaluation and verification |

See [TEAM.md](TEAM.md) for project roles.
