# Reply Coach

Dual-style reply coach for exchange students handling **casual peer coordination**.

It takes an English–Mandarin/Singlish message plus recent chat, then returns:

1. a **standard-English** reply
2. a **light local/Singlish** reply

This is the SMU CS425 group repo. The pipeline already runs so we are not starting from zero. It is **not** a finished project. Keep the repo **private**.

---

## Start here (everyone, today)

1. Clone this private repo. Do not fork it as public.
2. Put your name in [`TEAM.md`](TEAM.md). One person keeps the same role (A–E) for the whole project.
3. Read your role file under `roles/*/START_HERE.md`.
4. Run the setup below and confirm it works on your machine.
5. Help rewrite the 20 pilot items in `data/pilot/pilot_20.jsonl`. Those are drafts so the code can run. They are **not** gold data.

### Setup (Windows / PowerShell)

Use Python 3.10 or newer. Run all commands below from the project root (the folder
containing this README and `pyproject.toml`). Install dependencies into the
project's `.venv`, not another application's Python environment.

Run this on first setup, and rerun it after pulling dependency changes. An existing
environment is reused rather than recreated. The `ensurepip` step also handles
environments created without pip, such as the existing uv-created environment.

```powershell
$env:PYTHONUTF8 = "1"
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    python -m venv .venv
}
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -e ".[demo,dev]"
```

All commands below explicitly use that same environment. Activation is optional;
you do not need to change PowerShell's execution policy.

### Run persona judging — one command

After the Python setup above, open Ollama and install the standard local judge,
**Gemma4 12B**. No API key or hosted judge is used. The ignored `.env` can contain
`OLLAMA_MODEL=gemma4:12b` and `OLLAMA_BASE_URL=http://127.0.0.1:11434`.

```powershell
ollama pull gemma4:12b
.\.venv\Scripts\python.exe scripts/run_judging.py
```

Local Ollama judging explicitly disables thinking and uses a 4,096-token context
with up to 2,048 output tokens. These settings are saved with each run; changed
settings require a fresh run, not resuming older ratings. The model's installed
digest is recorded too. These runtime settings do not establish judge reliability.

This checks the setup, exports BM25 M0/M2 replies, prepares the same five
validation conversations for both settings, and runs every persona in
[`data/evaluation/personas.csv`](data/evaluation/personas.csv). With the initial
ten personas that means **200 judgments**. No manual export commands or input
folder are needed. Missing default `.env` is copied from `.env.example` without
overwriting an existing file; Python packages and Ollama models are not installed
automatically.

The terminal shows average appropriateness, naturalness, style fit and simulated
would-send percentage by setting/style. Each invocation creates a fresh ignored
folder under `experiments/judging/run_...`, prints its location and a ready-to-copy
resume command. The CSV is at `RUN_FOLDER/packet/ratings/summary.csv`.

Add `--check-only` to preview without file writes or judge requests. To continue
an interrupted judging stage, use `--run-dir RUN_FOLDER --resume`; to inspect
saved results, use `--run-dir RUN_FOLDER --summary-only`. The wrapper never
regenerates replies on resume. An interruption before packet preparation requires
a fresh run. Optional new-run flags: `--items`, `--methods bm25 minilm` and
`--memories M0 M1 M2`. Adding MiniLM can download its public embedding weights on
first use. Frozen test data is never used. These are AI ratings, not human survey
results; inspect pilot reliability before the main evaluation.

### Judge checks and interpretation

Start small using the **first existing CSV persona** on the same five validation
conversations, two BM25 memory settings and two styles: **20 judgments**, not 200.

```powershell
.\.venv\Scripts\python.exe scripts/run_judging.py --calibrate --check-only
.\.venv\Scripts\python.exe scripts/run_judging.py --calibrate
```

Calibration prints each conversation, reply, scores and explanation in readable
form. Inspect whether the judge catches wrong intents/contradictions and correctly
recognises good replies; valid JSON is not a quality check. Omit `--calibrate` only
when ready for the full persona panel. Do not treat a one-persona diagnostic as
the main experiment. The printed resume command preserves calibration mode.

Use the same Gemma model/digest, settings, rubric, personas and conversations for
every compared system. This controls the comparison conditions; it does not
eliminate judge bias or make ten personas independent human reviewers. Treat
scores as comparative AI judgments, supported by manual spot-checks, not actual
human feedback or calibrated accuracy. The pilot showed overly generous printer
suggestions and occasional local-style/persona bias; report these limitations.
Keep judge inputs anonymous even though evaluation stays local.

### Run checks

```powershell
.\.venv\Scripts\python.exe scripts/week4_smoke.py
.\.venv\Scripts\python.exe -m replycoach.demo_cli "eh later go makan? can 吃饭 then library"
.\.venv\Scripts\python.exe -m pytest -q
```

### Run the demo

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/replycoach/demo_app.py
```

Streamlit prints the local URL to open, normally `http://localhost:8501`. Stop the
server with Ctrl+C before restarting it. If you see
`MiniLM unavailable: No module named 'sentence_transformers'`, rerun setup above
and restart using this explicit `.venv` command, not plain `python` or `streamlit`.

The demo's **Compare reply suggestions** button runs BM25 (word matching) and
MiniLM (embedding similarity) side by side. Each method returns a Standard English
and a Light local suggestion: four clickable replies that fill the display-only
chat composer. They select saved replies; they do not generate new text. The CLI
continues to return one pair from BM25.

The default reply bank is still the **20 pilot drafts**, not DailyDialog or CoSEM.
The reviewed split is now published: **190 train/reply-bank items, 50 validation
queries and 60 held-out test items**. Choose **Reviewed development file** under
**Reply bank settings** and enter `data/dev/train.jsonl` (or an absolute path).
Every record must be `dev`, `completed`, explicitly privacy-safe,
and independently reviewed. Test, unassigned, incomplete and duplicate records
are rejected; selecting a file never edits or splits it.

MiniLM runs locally and may download public model weights on its first comparison.
The model and bank-specific vectors are reused across reruns. If MiniLM is
unavailable, BM25 remains visible and the demo offers a retry; it never substitutes
BM25 under the MiniLM heading. Both methods use the same selected memory,
normalization, reranker and fallback policy. **Ranking scores are not accuracy
percentages and are not directly comparable across methods.**

### Role E system modes and teammate handoffs

Under **System settings**, the default remains **Retrieval comparison** (BM25 vs
MiniLM). You can also select **Retrieval only**, **Generator only**, or **Hybrid**.
The single-system modes return two clickable replies. Choose BM25/MiniLM for
retrieval and V4/V5 for generation; resources load only when requested.

Role C's **task-trained V4 provider is connected**; its checkpoint must be copied
privately into `artifacts/rnn_v4_task.pt` or selected with
`REPLYCOACH_V4_CHECKPOINT`. See [C's handoff](roles/C_rnn_attention/HANDOFF.md)
for measured results and substantial quality limitations. V5 remains pending.
Missing V4/V5 models name Role C/D and fall back honestly:
partial hybrid runs are labelled retrieval-only, not successful hybrids. The
registry never uses the one-batch smoke checkpoint or fake production replies.

Search **`TODO: FOR ROLE`** in the repository (VS Code: Ctrl+Shift+F). Exact A/B/C/D
handoffs, provider contracts and artifact instructions are in
[`docs/role_handoffs.md`](docs/role_handoffs.md). The 20-case proof checklist is in
[`docs/demo_checklist.md`](docs/demo_checklist.md); it is not a completed run.

Validate the published disjoint development files without loading any model:

```powershell
.\.venv\Scripts\python.exe scripts/export_dev_outputs.py --check-only
```

Retrieval-only diagnostics do not require a generator. Once C/D connects a real
provider, export all three modes on the same inputs and cached candidate pools:

```powershell
.\.venv\Scripts\python.exe scripts/export_dev_outputs.py --mode retrieval_only --output experiments/retrieval_dev_run1.jsonl
.\.venv\Scripts\python.exe scripts/export_dev_outputs.py --generator v4 --mode all --output experiments/three_modes_dev_run1.jsonl
```

Use `--queries`/`--bank` for other development paths, `--retrieval bm25|minilm`,
and `--memory M0|M1|M2`. Exports reject held-out/unassigned rows, query-bank overlap,
incomplete modes and existing outputs; choose a new filename each run. Exact input
overlap is checked, but Role A must still group semantic near-duplicates. These
are development outputs, **not quality ratings or final held-out results**.
The interactive JSON download also records incomplete states and is not a benchmark.

### Evaluation plan updated 17 October 2026

Following the professor's approval reported by Jereme, the planned evaluation
replaces human output raters with **10 initial configurable persona-based AI
reviewers**, not three specialist judges. Each keeps the same persona and rates
every compared reply from the same conversation set, with system names hidden.
The three criteria are **appropriateness, naturalness and style fit**, each 1–5,
with a short explanation and problem/uncertainty flags. Naturalness replaces the
older proposal's usefulness criterion.

This is **persona-based LLM evaluation**, not a human survey or ten independent
human participants. Jereme/Role E now owns the judging workflow as well as the
output handoff; Tristan/Role A retains the frozen data split. The runner is
`scripts/judge_replies.py`, using local Gemma4 12B through Ollama and the common criteria. Existing spreadsheet
scripts remain optional human-review tools. See
[`docs/split_and_rating_protocol.md`](docs/split_and_rating_protocol.md)
for the protocol, safeguards and remaining work. Normally use the one-command
pipeline above; the manual stages below are only for custom exports/debugging.

<details>
<summary>Advanced: run individual export, packet and judging stages manually</summary>

Your Role E handoff is available through `scripts/prepare_judging_inputs.py`.
It validates matched development exports and creates **`judge_inputs.jsonl`** plus
**`PRIVATE_mapping.jsonl`** in a new folder. Keep both files locally; the AI
judges must receive only the public inputs, never the private mapping or the
whole folder. The default documented output location `experiments/judging/` is
ignored by Git. This tool makes no model calls and produces no ratings.

After exporting the BM25 run above, prepare a five-conversation pilot:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_judging_inputs.py --system BM25_M2 experiments/retrieval_dev_run1.jsonl retrieval_only --check-only
.\.venv\Scripts\python.exe scripts/prepare_judging_inputs.py --system BM25_M2 experiments/retrieval_dev_run1.jsonl retrieval_only --items 5 --out-dir experiments/judging/validation_pilot_01
```

Repeat `--system LABEL PATH MODE` for additional systems; paths containing spaces
can be quoted, including Windows absolute paths. Each selected mode must cover
all 50 validation queries before a common pilot subset is chosen. Omit `--items`
for all 50. Use a fresh output folder for every run. The local runner fans each
anonymous candidate out to the persona panel; this preparation step does not
create tool-using agents. Full field definitions and joining instructions are in the
protocol linked above.

Run the local judges on that prepared packet:

```powershell
if (!(Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe scripts/judge_replies.py experiments/judging/validation_pilot_01 --check-only
.\.venv\Scripts\python.exe scripts/judge_replies.py experiments/judging/validation_pilot_01
```

Set `OLLAMA_MODEL=gemma4:12b` in your ignored `.env`; the model must already
be installed and Ollama running locally. Edit/add rows in
[`data/evaluation/personas.csv`](data/evaluation/personas.csv) to change the panel.
Each persona rates every anonymous reply using the same rubric, including a
boolean `would_send`. The terminal shows per-model-setting and per-style averages
for appropriateness, naturalness, style fit, and simulated would-send percentage.
It saves `ratings/ratings.jsonl`, `run.json` and `summary.csv` inside the packet.
The five-conversation/two-setting example has 20 replies, hence 200 judgments.

The model-facing rubric is [`data/evaluation/judge_rubric.md`](data/evaluation/judge_rubric.md).
It contains only judging instructions and score anchors, not the administrative
notes in the optional human-review rubric. Context and requested style take
priority over persona habits. After a prompt change, first inspect a few good and
bad replies with one unchanged persona; successful JSON responses alone do not
validate the judge. Keep diagnostic runs separate from matched system results.

Add `--resume` after an interruption; successful judgments are not repeated.
Use `--summary-only` to display saved results without model calls. Use a fresh
`--out-dir` if the model, personas, rubric, script or packet changes. Failures stay
visible and are not given fabricated scores; incomplete summaries are provisional.
These numbers are simulated ratings, not proof of actual human preference or
accuracy. Pilot-check Qwen's judgments before treating them as experiment evidence.

</details>

The exception is evaluation only. Dataset authorship and independent human peer
review remain unchanged; AI ratings must not become gold replies or training
labels. ReplyCoach itself continues to run without hosted LLM generation APIs.

### Dataset authoring and peer review

To write original team dataset items without editing JSONL manually:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/replycoach/author_app.py
```

Choose your own name and role in the app. It automatically loads your assigned
author file, `data/authoring/items_<ROLE>.jsonl`, and saves your records there as
unassigned drafts.

To review your assigned teammate's items:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/replycoach/team_review_app.py
```

Choose your own role and name; the app automatically loads your assigned author's
file. Each item gets one independent peer review. The review cycle is A reviews B,
B reviews C, C reviews D, D reviews E, and E reviews A. Saving a valid review
automatically sets that item's `status` to `completed`. Role A assigns the fixed,
balanced split only after all 300 items are completed: 240 development and 60
held-out test.

Expected: 20 items load, BM25 returns a lunch pair, V3 trains one CPU batch if PyTorch is installed.

**Hard rules**

- No LLM-generated dataset items, gold replies or training labels. Evaluation-only
  AI ratings are permitted under the protocol above; they are not training data.
- No hosted LLM generation or judging APIs. Persona judging uses local Ollama
  with anonymised, privacy-checked inputs under the protocol above.
- Do not invent Mandarin. Do not force slang.
- Do not train on or tune with the held-out test set.

---

## Week 4 Friday proof (18 Sep 2026)

Each person shows a **runnable artefact**, not a progress update.

| Role | Who | Friday artefact |
| --- | --- | --- |
| A | Data and evaluation | Chair the 20-item rewrite (privacy + style). Save the reviewed JSONL and say how many passed. |
| B | Retrieval and memory | Run 5 queries. Save top-3 retrieved pairs under `experiments/logs/`. |
| C | RNN and attention | Run `.\.venv\Scripts\python.exe -m replycoach.generate.train_v3`. Share `artifacts/rnn_one_batch.json`. |
| D | Transformer experiment | Run `notebooks/D_mt5_kaggle_smoke_test.ipynb` on a **Kaggle GPU** (no API). Put `mt5_smoke_report.json` in `experiments/logs/`. |
| E | Reranker, demo | Run CLI + Streamlit. Screenshot STANDARD + LOCAL. |

Fill names in `TEAM.md`. First come, first served, then freeze.

---

## Week 7 — data and review cycle

Code is ahead of data. Each member authors **60 unassigned items** and independently peer-reviews one teammate’s set: A reviews B, B reviews C, C reviews D, D reviews E, and E reviews A. Each member reviews 60 items.

The five role files produce **300 items**. Once every item has a valid review and is marked `completed`, Role A assigns one fixed, balanced 80:20 split: **240 development + 60 untouched test**.

Use [`data/authoring/worksheet.md`](data/authoring/worksheet.md) and [`docs/annotation_guide.md`](docs/annotation_guide.md).

On the side:

- **A** — duplicates, privacy, style consistency, split labels
- **B** — V1 lexicon edits in `data/lexicons/local_terms.json`
- **C** — train V3 on DailyDialog public pairs (not the test set)
- **D** — `STANDARD` / `LOCAL` input format; development data only
- **E** — clearer retrieved-candidate display in the demo

Authors do not choose development or test records. Anything exposed during model development must remain outside the held-out test set.

---

## Dates

| When | What |
| --- | --- |
| 18 Sep 2026 | Week 4 Friday artefacts |
| 29 Sep 2026 | Pre-proposal workshop, ~15 min. E runs the live demo. Outline: `docs/preproposal_workshop.md` |
| 2 Oct 2026 | 3-page proposal. Draft: `docs/proposal_draft.md` |
| 10 Nov 2026 | Final presentation |
| 27 Nov 2026 | Report, code, data on eLearn |

---

## What this system is

Incoming message + dynamic session memory  
→ V1 detect/normalise English–Mandarin/Singlish terms  
→ V0/V2 retrieve similar vetted reply pairs  
→ V4 or V5 generate extra candidates (keep only the better generator)  
→ V6 rerank  
→ confidence fallback  
→ two suggestions: STANDARD and LOCAL

**Locked decisions**

- Scope is casual peer coordination only.
- V4 = V3 RNN + attention. V5 (mT5-small on Kaggle) is compared with V4; we keep only the winner.
- Memory is dynamic: newest two turns plus up to two relevant older turns from the last eight session messages.
- Local style is restrained and English-led.
- Unknown/low-confidence inputs fall back to safe neutral English.

**Corpus**

- Dual-style task data = **team-authored, independently human-reviewed** items;
  the official 300-item split is published. The demo still defaults to pilot drafts.
- Public **DailyDialog** (CC BY-NC-SA 4.0) is for generic V3/V4 dialogue pretraining only. It is not downloaded yet and does not replace STANDARD/LOCAL labels.
- V5 uses local `google/mt5-small` weights on Kaggle, not an API.

`data/pilot/pilot_20.jsonl` is an engineering fixture (`draft_needs_team_review`). Rewrite and peer-review it before treating it as gold.

**CS425 constraints:** train/run ReplyCoach on a PC, Colab, or SCIS GPU; no hosted
LLM generation APIs inside ReplyCoach; there must be a new contribution, not a
copy of an open-source chatbot. The professor-approved exception reported by
Jereme permits LLMs as offline evaluators only, subject to the protocol above.

More detail: `docs/architecture.md`, `docs/licence_checklist.md`, `docs/week_plan.md`.
