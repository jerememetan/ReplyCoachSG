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

Open [http://localhost:8501](http://localhost:8501). The app opens in the final
pipeline with **M2** memory and the **Reviewed development file**
(`data/dev/train.jsonl`, 190 examples). Set up the presentation checkpoints
below before generating replies.

## Run the full presentation demo

Download the [demo checkpoint archive](https://drive.google.com/file/d/15UAlPPn2LGflhzIoQwglMkDf6mtldctn/view?usp=sharing)
(`ReplyCoachSG-demo-checkpoints.zip`, 7.68 GB). Extract it from the repository
root (the folder containing `pyproject.toml`). On Windows PowerShell, run the
following command from that folder.

```powershell
tar.exe -xf "$env:USERPROFILE\Downloads\ReplyCoachSG-demo-checkpoints.zip"
```

Adjust the archive path if you saved it somewhere else. The archive already
contains the `artifacts/` directory structure. If you received the checkpoints
individually, place the **complete** files or folders at these locations:

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

1. Enter the **Recent conversation** and **Incoming message**. Use one turn per
   line in the conversation, prefixed with `You:` or `Peer:`.
2. Click **Suggest two replies**, then click a reply to preview it in the composer.
3. Open **Details** to inspect the model proposals and the Qwen scores under
   **Local wording candidates**.

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
Local reply. V7 v3 tries edits to different phrases, including particle forms
such as `can lah` and `okay lor`. Qwen ranks up to six distinct variants
alongside the original whenever at least one is available. With zero variants,
the app keeps the original. The comparison
table also shows BM25, MiniLM and V4 proposals. If SEA-LION fails to load, the
table can still show available proposals and the app reports the model error.
[V7 v3 method and fallback behavior](docs/v7_random_lexical.md)

## Try retrieval without the presentation checkpoints

After installing `.[demo]`, launch the separate comparison app. On Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/replycoach/demo_experiments.py --server.fileWatcherType none
```

On macOS or Linux, use `./.venv/bin/python`. The app opens in **Retrieval
comparison** with BM25 and MiniLM. MiniLM may need internet access for its
public embedding model on first use. For BM25 alone, choose **Retrieval only**
and select **BM25**. This entry point also keeps the experimental controls for
comparing generators, memory settings, and reply banks.

## Results

| Final demo output | Appropriateness /5 | Naturalness /5 | Style fit /5 | Would send |
| --- | ---: | ---: | ---: | ---: |
| Standard · V5 SEA-LION | 4.555 | 4.975 | 5.000 | 88.0% |
| Light Local · V7 v3 | 4.655 | 4.948 | 4.393 | 86.5% |

These are simulated ratings from ten Gemma personas on 60 previously inspected
test conversations. The Local replay used one reproducible candidate draw per
conversation with the expanded particle mappings.
[V7 v3 evaluation and comparison with V7 v2](docs/evaluation/v7_v3_local_20261115.md)

- [Transformer selection on 50 validation conversations](docs/evaluation/transformer_selection_20261108.md)
- [Six-system evaluation including historical V7 v1](docs/evaluation/current_pipeline_20261108.md)
- [V6 refinement and memory experiments](docs/evaluation/qwen_refinement_20261113.md)
- [Follow-up Standard V6 fine-tuning](docs/evaluation/qwen_standard_continuation_20261113.md)
- [Architecture and version history](docs/architecture.md)
- [Evaluation entry points and data splits](docs/final_test_evaluation.md)

## Preprocessing, training, and evaluation

The prepared files in `data/dev/` and `data/test/` contain 190 training,
50 validation, and 60 test conversations. The demo uses the training file as
its reply bank. The test split is used for evaluation.

- **Regenerate the split.** Use the [pinned research snapshot](https://github.com/jerememetan/ReplyCoachSG/tree/8c50579fc675f06db2051bdaa58345a82208e2ad)
  in a separate folder. From its root, with the project dependencies installed,
  run `python scripts/data-split/make_split.py --items-dir data/authoring --out-dir .`
  with that environment's Python. The included manifest fixes the assignments.
- **Training.** The research snapshot contains the [RNN training entry point](https://github.com/jerememetan/ReplyCoachSG/blob/8c50579fc675f06db2051bdaa58345a82208e2ad/scripts/run_role_c.py),
  [RNN reproduction instructions](https://github.com/jerememetan/ReplyCoachSG/blob/8c50579fc675f06db2051bdaa58345a82208e2ad/roles/C_rnn_attention/EXPERIMENT_REPORT.md#reproduction),
  [V6 Qwen training notebook and input-bundle instructions](https://github.com/jerememetan/ReplyCoachSG/blob/8c50579fc675f06db2051bdaa58345a82208e2ad/docs/dailydialog_reranker_training.md#run-it-on-kaggle),
  and the [SEA-LION adaptation recipe](https://github.com/jerememetan/ReplyCoachSG/blob/8c50579fc675f06db2051bdaa58345a82208e2ad/roles/D_transformer/sealion_repair25_MODEL_CARD.md).
  Exact retraining requires the original training inputs and starting checkpoints.
  The demo checkpoint archive supplies the inference artifacts.
- **Evaluation.** The [evaluation guide](docs/final_test_evaluation.md) documents
  `scripts/run_final_test.py --current-pipeline`, which runs the historical
  six-system comparison with V7 v1. The [V7 v3 evaluation](docs/evaluation/v7_v3_local_20261115.md)
  reports the subsequent Local-only replay. Re-running that follow-up requires
  the saved SEA-LION outputs, the matching Gemma judge, and the replay inputs.

## Repository map

| Path | Contents |
| --- | --- |
| [`src/replycoach/`](src/replycoach/) | Demo, retrieval, generation and ranking code |
| [`data/`](data/) | Reply bank, evaluation splits, lexicons and rubric |
| [`docs/evaluation/`](docs/evaluation/) | Consolidated findings and aggregate results |
| [`scripts/`](scripts/), [`tests/`](tests/) | Selected evaluation utilities and demo tests |

See [TEAM.md](TEAM.md) for project roles.

Historical evaluation replays require the original checkpoints and saved judging packets from the project team.
