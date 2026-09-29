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

```powershell
$env:PYTHONUTF8 = "1"
python -m pip install -e ".[demo,dev]"
python scripts/week4_smoke.py
python -m replycoach.demo_cli "eh later go makan? can 吃饭 then library"
python -m streamlit run src/replycoach/demo_app.py
pytest -q
```

To write original team dataset items without editing JSONL manually:

```powershell
python -m streamlit run src/replycoach/author_app.py
```

Choose your own name and role in the app. It automatically loads your assigned
author file, `data/authoring/items_<ROLE>.jsonl`, and saves your records there as
unassigned drafts.

To review your assigned teammate's items:

```powershell
python -m streamlit run src/replycoach/team_review_app.py
```

Choose your own role and name; the app automatically loads your assigned author's
file. Each item gets one independent peer review. The review cycle is A reviews B,
B reviews C, C reviews D, D reviews E, and E reviews A. Saving a valid review
automatically sets that item's `status` to `completed`. Role A assigns the fixed,
balanced split only after all 300 items are completed: 240 development and 60
held-out test.

Expected: 20 items load, BM25 returns a lunch pair, V3 trains one CPU batch if PyTorch is installed.

**Hard rules**

- No LLM-generated dataset items or labels (no ChatGPT/Cursor gold replies).
- No hosted LLM/API calls in the system.
- Do not invent Mandarin. Do not force slang.
- Do not train on or tune with the held-out test set.

---

## Week 4 Friday proof (18 Sep 2026)

Each person shows a **runnable artefact**, not a progress update.

| Role | Who | Friday artefact |
| --- | --- | --- |
| A | Data and evaluation | Chair the 20-item rewrite (privacy + style). Save the reviewed JSONL and say how many passed. |
| B | Retrieval and memory | Run 5 queries. Save top-3 retrieved pairs under `experiments/logs/`. |
| C | RNN and attention | Run `python -m replycoach.generate.train_v3`. Share `artifacts/rnn_one_batch.json`. |
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

- Dual-style task data = **team-authored** items only (pilot drafts now; official set still to write).
- Public **DailyDialog** (CC BY-NC-SA 4.0) is for generic V3/V4 dialogue pretraining only. It is not downloaded yet and does not replace STANDARD/LOCAL labels.
- V5 uses local `google/mt5-small` weights on Kaggle, not an API.

`data/pilot/pilot_20.jsonl` is an engineering fixture (`draft_needs_team_review`). Rewrite and peer-review it before treating it as gold.

**CS425 constraints:** train/test on a PC, Colab, or SCIS GPU; no LLM APIs; there must be a new contribution, not a copy of an open-source chatbot.

More detail: `docs/architecture.md`, `docs/licence_checklist.md`, `docs/week_plan.md`.
