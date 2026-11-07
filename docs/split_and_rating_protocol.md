# Fixed split and persona based LLM evaluation

## User quality requirements clarification

The primary requirements are recognizably natural Singlish for LOCAL,
approximately matching meaning across STANDARD/LOCAL, and coherent next-turn
responses even when they differ from the authored reference. See
[the requirements](model_quality_requirements.md). Near-neutral English no
longer qualifies for top LOCAL style marks merely because a Singaporean might
say it. Avoid artificial particle stuffing. The current one-draft persona schema
cannot measure cross-style meaning; use an additional paired review. Original-
reference token loss alone is insufficient to select a winner. Preserve prior
frozen reports; future evaluations must use fresh matched packets and record the
revised rubric. These changes do not create human gold labels or alter splits.

## Selectable-draft policy update — 3 November 2026

The project user reviewed six examples and permits plausible new personal or
factual details in drafts they select/edit before sending, including progress,
room, deadline and professor-approval replies. Missing prior evidence alone is
not an appropriateness error. Explicit contradictions, role reversals and missed
intent remain errors; agreeing is not automatically better than a valid refusal.
The evaluator-only rubric and shared rating anchors now reflect this policy.
Would-send is conditional on a user selecting a draft that fits their situation,
not a certification of unknown real-world facts.

Existing judge runs remain historical under their recorded rubric. Use fresh
calibration and fresh matched output folders after this prompt change; do not
merge them into old ratings. The runner records rubric text in the manifest and
rejects resumes after rubric changes. Evaluation persona ratings remain separate
from training labels; the fixed train/validation/test split is unchanged.

The earlier protocol below still supplies the same three score dimensions,
persona design and data guards. This update changes the interpretation of draft
facts, not the schema or completed-result status.

Updated 18 October 2026. Jereme reports professor approval to replace human output
raters with persona-based AI reviewers. The agreed design is a configurable panel
of **10 initial personas**, each scoring **appropriateness, naturalness and style fit**
(whole numbers 1–5). Naturalness replaces the original proposal's usefulness
criterion. This document supersedes older blind-human-rating instructions.

**Implementation status:** the frozen data files, guarded development exporter and
Role E's anonymous JSONL packet preparation exist. Jereme/Role E now also owns the
local persona runner, rubric/persona configuration, pilot checks and analysis.
`scripts/judge_replies.py` uses an editable CSV and the standard local Gemma4 12B
judge through Ollama. The team accepts this as a practical comparative evaluator,
not a validated replacement for independent human feedback. Use identical judge
settings and manual spot-checks, and report known contextual and persona/style
biases. The main evaluation and final test have not been completed. Tristan/Role A retains
data/split stewardship. Packet preparation itself still makes no model calls.
See sections 7–8 for the input and rating contracts.

## 1. What was done

- **Gate:** 300 items, 60 per author, all `completed`, all `privacy_ok`, review chain A<-E, B<-A, C<-B, D<-C, E<-D intact, no duplicate IDs.
- **Near-duplicates:** every pair scored on message + last two turns + STANDARD reply (`scripts/data-split/near_dups.py`).
  Pairs scoring >= 0.42 are grouped automatically; 6 more groups were added by hand
  (`data/split/review/manual_near_dup_groups.json`). A group is always kept on one side of every split.
- **Split:** `scripts/data-split/make_split.py`, seed 425, run once, seed not tuned. Group-aware simulated annealing to match
  proportions of author, scenario, intent, number of local terms and Mandarin/pinyin mix. Exactly 12 test items per author.
- **Validation:** 50 items inside the 240 dev items (10 per author), chosen the same way. Everything else in dev is the
  reply bank / training pool (190).

| file | items | `split` value | use |
|---|---|---|---|
| `data/dev/train.jsonl` | 190 | `dev` | reply bank, training |
| `data/dev/validation.jsonl` | 50 | `dev` | shared validation queries for all experiments |
| `data/test/test.jsonl` | 60 | `test` | locked; final evaluation only |

Record contents are unchanged except the `split` field. The frozen assignment is `data/split/manifest.json`. Re-running
`make_split.py` with the manifest present only re-applies it, so fixing an item's text never reshuffles the split.

## 2. Freeze checklist (do in this order)

1. Fix the flagged items in `data/authoring/items_*.jsonl` (see `data/split/review/audit_issues.csv`, errors first).
   Fix in place; do not delete items.
2. If audit corrections are agreed, `python scripts/data-split/make_split.py --items-dir data/authoring --out-dir .`
   re-applies the manifest; re-run `scripts/data-split/verify_split.py`. Do not casually change frozen data.
3. `python scripts/export_dev_outputs.py --check-only`
4. Commit, tag `data-v1`, and paste the SHA-256 values from `data/split/split_report.md` into the team chat.
5. Announce: nobody edits these files again; nobody inspects, tunes on or trains with `test.jsonl`.
   Consider keeping `test.jsonl` off the shared branch until the final evaluation.

Message for the team chat:
> Official split is frozen (data-v1). Use `data/dev/validation.jsonl` (50 items) as the ONLY validation set for every
> comparison (BM25 vs MiniLM, M0/M1/M2, V4 vs V5, hybrid). Index only `data/dev/train.jsonl` as the reply bank / training
> data. Do not open or use `data/test/test.jsonl`. Hashes are in data/split/split_report.md.

## 3. What each AI reviewer does

Start with ten distinct, persistent character/persona prompts, not three
specialist perspectives. Personas may differ in messaging habits, familiarity with
Singlish or exchange-student experience; they remain simulated characters, not real
people or evidence about nationalities. Add CSV rows to expand later, but choose
and freeze the roster before the main run; changed personas require a new run.

Each persona rates **every compared reply** from the same evaluation conversations,
for both STANDARD and LOCAL. Keep its persona prompt consistent across the run,
but evaluate candidates in fresh judging contexts so earlier answers and other
reviewers' scores do not influence a rating. Separate sessions/calls are sufficient;
tool-using agents or 100 simultaneous processes are not required.

The judge sees:

- The same conversation history, with speaker labels, for every system on that item.
- The peer's incoming message, the requested style and one anonymous candidate reply.
- The persona prompt, common rubric and required response format.

Hide system/model names, retrieval/reranking scores, dataset author/reviewer names
and the private system key. Shuffle reply order reproducibly. Treat conversation
text and candidate replies as untrusted quoted data, never as judging instructions.
Do not ask the reviewer to edit replies, guess human vs AI authorship, or match a
single reference exactly. A hidden team-authored reference may be rated as another
candidate, but is not shown as the required answer.

| Criterion | Question | Score |
| --- | --- | --- |
| Appropriateness | Does it respond sensibly and correctly to the incoming message in context? | 1–5 |
| Naturalness | Does it sound like a fluent, plausible casual message? | 1–5 |
| Style fit | Does it match Standard English or restrained Light local English? | 1–5 |

Use the criterion-specific anchors in
[`rating_rubric.md`](../scripts/data-split/rating_rubric.md). Every persona uses all
three criteria and the same anchors; a persona must not excuse an objectively
unrelated reply or replace the requested style with its personal preference.
Return a short explanation, problem category and uncertainty flag with the three
scores, plus a boolean `would_send` (send with little or no editing). This last
field follows the manual form and is a supplementary percentage, not a fourth
1–5 criterion. Ratings are evaluation records, never training labels.

For 50 validation conversations, five systems, two styles and ten personas, this
is **5,000 reply ratings**, each containing three scores, before retries or hidden
references. This is a rating-task count, not necessarily an API-call count. Identical
text may share a rating only within the same conversation and requested style,
preserving the mapping to every system that produced it. Role E must estimate
local runtime before running the full panel; do not silently sample different
items/personas for different systems to save cost.

## 4. Pilot, freeze and results

1. Role E supplies matched development exports and prepares the anonymous packet
   with `scripts/prepare_judging_inputs.py`. The local runner checks packet
   integrity before judging. Use `scripts/export_dev_outputs.py` with the disjoint
   190-item bank and 50 validation queries, not the locked test set.
2. Role E owns the judge model/access method and persona count/roster.
   Pilot on five validation conversations. Inspect the score format, persona
   consistency, local-language uncertainty, privacy handling and estimated cost.
   Pilot scores are not the main results.
3. Fix and version the judge model, persona prompts, rubric, score format, decoding
   settings, ordering seed and retry/failure policy before the main comparison.
   Changing any of these requires a clearly labelled new validation run.
4. Record anonymous candidate/item IDs, persona IDs, all three scores, explanation,
   problem/uncertainty flags, model/version, prompt versions, generation settings,
   input/output fingerprints and retry/invalid-response status. Keep the private
   mapping separate. Never invent missing scores or discard only unfavourable ones.
5. Report each criterion by system and style. First average valid persona ratings
   within each conversation/system/style, then summarise across conversations.
   Retain persona-level results and disagreements, and disclose missingness and
   uncertainty. Do not treat personas from one base model as independent
   human raters or multiply the effective number of evaluation conversations.
   If reporting uncertainty intervals, resample conversations, keeping their
   systems/styles/persona scores together for paired comparisons.
6. Optional team sanity checks of about ten validation cases, including disagreements,
   can identify obvious judge errors. They are not a large blind human survey and
   do not replace the dataset's independent human peer review. Report them separately.
7. Freeze the systems, judge setup and reporting rules before the final 60-item test
   evaluation. Coordinate once with A's frozen split; do not tune from test results.
   The development exporter intentionally refuses test inputs. A separate final-test
   output route remains to be implemented; do not remove the development guard.

Call the method **persona-based LLM evaluation**, not a human survey or actual user
study. Shared-model biases, correlated personas and uncertain cultural judgements
remain limitations, however large the panel. More persona ratings are not more
independently sampled conversations.

The professor-approved exception reported by Jereme is **evaluation only**.
ReplyCoach generation/training remains local without hosted LLM generation APIs.
Send only privacy-checked task text and outputs to the approved evaluator, never
credentials, identifiable chats, restricted handouts or repository files. No model
calls, paid judging or new training labels are authorised by this documentation.

## 5. Remaining implementation handoffs

- **Role E owns judging:** `scripts/judge_replies.py` implements local inference,
  schema validation, one retry, saved progress and per-setting/style summaries.
  Freeze the persona roster/config after pilot review; main evaluation, independent
  sanity checks and analysis remain work, not completed experimental results.
- **Role A owns data:** preserve the reviewed split/manifest and support independent
  pilot sanity checks and the later final-test handoff.
- **Ready for Role E:** export-to-judge preparation validates coverage and writes
  anonymous inputs plus a separate private mapping, without manual JSON edits.
  See section 7. **TODO: FOR ROLE E with A:** provide a separate guarded final-test
  output route; that route is not implemented yet.
- **TODO: FOR ROLE B/C/D:** deliver matched real-system outputs and settings on the
  shared validation set. Do not describe absent generators or degraded hybrids as
  completed experimental systems.

## 6. Optional legacy human spreadsheet tools

`scripts/data-split/make_rating_kit.py` and `analyze_ratings.py` implement the earlier
human-rating workflow. They do not spawn personas, call an LLM or analyse the new
AI panel. Do not put persona names into `--raters` and present those sheets as the
implemented AI evaluation. They may still support optional manual checks with the
same appropriateness/naturalness/style-fit rubric.

The old design used five teammates, excluded an item's author/reviewer, assigned
two eligible human raters per item, and included a hidden reference by default.
This assignment workload is **not required** for the new AI panel. For optional
calibration, all humans can rate five shared items; discard pilot scores and settle
rubric interpretation before the main run.

The spreadsheet tools require `openpyxl`, which is not currently declared in the
project's setup. These are legacy/manual commands, not commands for the future AI
runner. From the project root, an example with an existing retrieval export is:

```powershell
.\.venv\Scripts\python.exe -m pip install openpyxl
.\.venv\Scripts\python.exe scripts/data-split/make_rating_kit.py --queries data/dev/validation.jsonl --items 5 --calibration --system "BM25=experiments/retrieval_dev_run1.jsonl:requested_mode=retrieval_only" --standard-field standard.candidate_text --local-field local.candidate_text --raters "Tristan,Cody,Yangtian,Hao Ming,Jereme" --out-dir rating_kit_calib
.\.venv\Scripts\python.exe scripts/data-split/analyze_ratings.py --key rating_kit_calib/PRIVATE_rating_key.csv --sheets "returned/*.xlsx" --out rating_report
```

The kit uses relative export paths above; its legacy `NAME=FILE[:filter]` parser
does not support Windows drive-letter paths. The export's mode key is
`requested_mode`, and reply text is nested under `standard.candidate_text` and
`local.candidate_text`. Keep private keys, returned sheets and raw ratings out of
Git; existing ignore rules do not automatically cover arbitrary rating folders.
Use a fresh output folder and do not overwrite previous runs. A human-analysis
report must not be relabelled as persona-based LLM results.

## 7. Implemented development packet handoff

`scripts/prepare_judging_inputs.py` reads existing development exports. It never
loads a generator, calls a judge or creates scores/personas. The producer must
first export real system outputs with `scripts/export_dev_outputs.py`.

```powershell
.\.venv\Scripts\python.exe scripts/export_dev_outputs.py --mode retrieval_only --retrieval bm25 --memory M2 --output experiments/logs/bm25_m2_run1.jsonl
.\.venv\Scripts\python.exe scripts/prepare_judging_inputs.py --system BM25_M2 experiments/logs/bm25_m2_run1.jsonl retrieval_only --check-only
.\.venv\Scripts\python.exe scripts/prepare_judging_inputs.py --system BM25_M2 experiments/logs/bm25_m2_run1.jsonl retrieval_only --items 5 --seed 425 --out-dir experiments/judging/validation_pilot_01
```

Repeat `--system LABEL PATH MODE` to compare more systems. Labels must be unique;
the mode is explicitly `retrieval_only`, `generator_only` or `hybrid`. If an export
contains all three modes, only the chosen mode is selected. Three separate
arguments avoid the legacy spreadsheet parser's Windows drive-letter problem.
Quote paths/labels containing spaces. The full selected mode must cover every
validation item exactly once before sampling a common `--items 5` pilot. Omit
`--items` to use all validation conversations.

The command rejects query/bank overlap, test/unassigned/unreviewed/unsafe datasets,
wrong input fingerprints/messages, incomplete/extra/duplicate selected query rows,
mixed run settings within one system, failed/degraded modes and malformed
candidates. It preserves real low-confidence fallback replies for evaluation;
they are not silently removed to improve results. Different labelled systems
may intentionally vary retrieval or memory settings. It does not add reference
replies or deduplicate identical outputs automatically.

### Public file contract

Give the judges **only `judge_inputs.jsonl`**. Each line is one anonymous candidate
to score, not an existing score and not one persona. The local runner applies each
persona to every candidate. Fields are:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1`; reject unsupported versions. |
| `candidate_id` | Anonymous ID such as `C000001`, assigned after shuffling. |
| `conversation_id` | Anonymous ID shared by all systems/styles for the same conversation. |
| `conversation` | Original prior turns in chronological order, each `{speaker, text}`; consecutive same-speaker turns remain intact. |
| `incoming_message` | `{speaker: "peer", text: ...}` for the message being answered. |
| `style` | `STANDARD` or `LOCAL`, the requested reply style. |
| `reply` | Exact suggested candidate text; not the authored gold answer. |

There are no original item IDs, system names, scores, author/reviewer details,
paths, model metadata or gold-answer fields in the public record. Conversation
history is the same across systems, even when their selected memory differs.
This blinds metadata; it is not automatic personal-data redaction or a guarantee
that a reply's wording cannot reveal its source. E must still inspect text/privacy.

### Private mapping contract

`PRIVATE_mapping.jsonl` has one corresponding record per public `candidate_id`.
It contains `schema_version`, `candidate_id`, `conversation_id`, original `item_id`,
`style`, `system` label, `requested_mode`, `memory_mode`, `retrieval_method`,
`source_export` label and its raw-byte `source_export_sha256`, the bank and query
fingerprints, `shuffle_seed`, `conversation_count`, and the complete original
selected `candidate` including scores/provenance/decoding metadata. Every saved
mapping also records `judge_inputs_sha256`, the SHA-256 of the public file's actual
bytes. Keep this file local to the team; never include it in a judge prompt or
upload the entire packet folder to the evaluator. `PRIVATE` is a convention, not
filesystem access control. `source_export` is a forward-slash repository-relative
path for an export inside the repo, or just the filename for an external export.
It never stores the machine's absolute directory path; the hash identifies the
exact source contents even when filenames are the same.

Both files are UTF-8 JSONL and must remain together unchanged for analysis. IDs are
unique **within a packet**, not across runs. The local runner must:

1. Require both files and verify the public file hash, schema version and exact
   one-to-one candidate-ID mapping before any judging call. This also rejects
   partial packets after an unexpected write failure.
2. Iterate the public records for each frozen persona. Attach the packet hash,
   `candidate_id` and `persona_id` to each returned rating, plus judge config/run
   metadata. Require the agreed three integer scores and explanations/flags.
3. Join ratings to systems using the private mapping only during local analysis.
   Never join different packets by candidate ID alone or treat duplicate IDs as
   duplicate ratings across separate runs. Preserve pairings for conversation-level
   comparisons. E owns response validation, retries, persona configuration and
   aggregation; these are implemented in the separate judging script, not the
   packet preparation tool.

Use a new output folder every time; existing directories (even empty ones) and
targets under `data/` are refused. Validation/serialization failures create no
packet. Unexpected disk errors may leave an incomplete directory: do not use it
or overwrite it; inspect it and choose a fresh run directory. The command reports
success only after both files are published. `--check-only` writes nothing.
`experiments/judging/` is ignored by Git; custom output locations are not
automatically private or ignored. No test-split override exists in this command.

## 8. Minimal local judging command

Use `.env.example` as the setup template; keep `.env` ignored. Simple `KEY=value`
entries select `OLLAMA_MODEL=gemma4:12b` and
`OLLAMA_BASE_URL=http://127.0.0.1:11434`. These are also the defaults. Existing
environment variables override the file. No new dependencies, provider selector
or API key are needed. Open Ollama and install with `ollama pull gemma4:12b`.
This route accepts loopback HTTP only, requires an installed local model and never
downloads a model or substitutes an Ollama cloud model. Uses Ollama's
[chat API](https://docs.ollama.com/api/chat) and
[JSON-schema outputs](https://docs.ollama.com/capabilities/structured-outputs).

```powershell
.\.venv\Scripts\python.exe scripts/judge_replies.py experiments/judging/validation_pilot_01 --check-only
.\.venv\Scripts\python.exe scripts/judge_replies.py experiments/judging/validation_pilot_01
.\.venv\Scripts\python.exe scripts/judge_replies.py experiments/judging/validation_pilot_01 --resume
.\.venv\Scripts\python.exe scripts/judge_replies.py experiments/judging/validation_pilot_01 --summary-only
```

The first command checks files without contacting Ollama or creating ratings.
The second runs the form for every candidate/persona. The third skips completed
judgments and retries failed ones; the last displays saved summaries offline.
Use `--out-dir` for a different fresh run folder. One process runs sequentially;
another process cannot write the same run simultaneously. No new Streamlit app.

Edit `data/evaluation/personas.csv`: columns `id,description`, initially ten unique
fictional student profiles. All use the same shared rubric and fresh contexts.
There are no three specialist judges or different scoring scales by persona.
Twenty candidates with ten profiles means 200 judgments, not 200 model copies.

The standard local configuration explicitly disables thinking, uses a 4,096-token
context and allows up to 2,048 output tokens, with temperature 0 and seed 425.
Record the installed Gemma model digest and these exact settings in every run.
The runtime disables proxies and redirects; no hosted judge or credentials are
supported. Neither malformed nor truncated responses receive fabricated scores.

All systems must use the same Gemma configuration, rubric, persona set and
conversations. This makes conditions controlled and consistent, not necessarily
unbiased. Pilot inspection found overly generous responses to the printer
operation question, assumptions about sharing notes that were not in the input,
and occasional persona habits influencing style/appropriateness scores. Keep
these limitations in the report and manually spot-check a selection from each
system/style, including both good and bad replies. Do not treat shared-model
persona agreement as independent validation or claim the ratings equal human
feedback. Use the three criteria separately rather than hiding weaknesses in a
single combined score. Other local judge models remain configurable for explicit
experiments, but do not mix judges/settings within a comparison.

Use `--calibrate` on either judging command to retain all selected conversations
and both styles but only the first existing persona. The normal five-conversation,
two-setting packet therefore takes 20 judgments. Output is explicitly diagnostic,
with readable context, replies, scores and explanations; it is not the main
evaluation. Keep `--calibrate` on resume (the wrapper prints it). A fresh full-panel
run is required after calibration; do not combine different persona coverage.

The runner loads the concise, evaluator-only anchors from
`data/evaluation/judge_rubric.md`; it no longer sends the administrative notes or
optional human-review instructions in `scripts/data-split/rating_rubric.md` to the
model. The three criteria and response schema are unchanged. Established facts,
incoming-message intent and the requested style outrank personal texting habits.
The exact AI rubric is saved in `run.json` and participates in resume checks.
After changing a prompt, first diagnose a few contextually good and bad replies
with one unchanged persona. Keep those checks labelled diagnostic, separate from
matched system summaries; do not turn them into training labels or tune on test
items. Then check additional validation cases not used to revise the prompt.

The exact model response schema is enforced in `scripts/judge_replies.py`:

```json
{
  "appropriateness": 4,
  "naturalness": 4,
  "style_fit": 5,
  "would_send": true,
  "explanation": "Answers the question naturally in the requested style.",
  "problem": "none",
  "uncertain": false
}
```

All three scores are integers 1–5 (not booleans, strings or floats). `would_send`
and `uncertain` are JSON booleans. Explanations are nonempty and at most 500
characters. Problem categories are `none`, `garbled_or_unreadable`,
`offensive_or_unsafe`, `personal_info`, `generic_or_unrelated_fallback`.
Reject extra/missing fields. Each judgment retries once on malformed responses
or request errors, then records a failure with no fabricated scores.

Outputs in `PACKET/ratings/` are `run.json` (frozen settings), `ratings.jsonl`
(append-only persona ratings/errors/attempts/timing) and regenerated `summary.csv`.
The run binds both packet files, persona CSV, script, rubric, schema, provider,
calibration mode and generation settings, including the installed local model
digest and explicit thinking flag. Changed settings cannot resume an old run.
Saved summaries remain readable offline after a script change, but a new judging
run is required; do not silently rewrite old run settings or ratings.
Keep all these files ignored and local. An interrupted malformed final JSONL line
requires inspection/recovery from a backup, not automatic destructive truncation.

The terminal refreshes after each candidate. Each model setting and requested
style gets its own row: appropriateness, naturalness, style fit, would-send %, and
valid/expected counts plus failures. First average personas within each
conversation/system/style, then average conversations equally. Would-send % uses
the same weighting; with complete coverage it is `100 * yes / all judgments`.
Failures are missing, not zero scores or no votes. Incomplete rows are explicitly
provisional; do not declare a winner when coverage differs. No combined score or
automatic winner is invented. A complete matched run is still simulated evidence,
not actual human preference, population statistics or proof that replies are human.
Gemma4 12B is the project's standard local comparative evaluator, not a validated
Singlish expert or substitute for independent human participants.

## 9. One-command pipeline (normal entry point)

After the normal Python project setup, configure the local judge in `.env`.
Ollama and the installed Gemma4 12B model must be ready. Then use:

```powershell
.\.venv\Scripts\python.exe scripts/run_judging.py
```

This thin wrapper uses the same Python executable and existing scripts: validate
the development split/personas/judge configuration, export BM25 M0 and M2 on all validation
queries, prepare the same five sampled conversations, then judge their 20 replies
with all ten CSV personas. Default workload: 200 judgments. It does not recreate
or inspect the locked test split, download judge models, install dependencies or
implement another scoring system. A missing default `.env` is copied exclusively
from `.env.example`, never overwriting an existing config.

Every fresh run uses a new ignored `experiments/judging/run_TIMESTAMP/` containing
`exports/`, `packet/` and `packet/ratings/`. There is no requirement to receive
someone else's generated packet folder. The wrapper prints the result path and
an exact resume command before judging. Select an explicit fresh folder with
`--run-dir`. Methods/memories/items are new-run options, defaulting to BM25,
M0/M2 and five items; `--methods bm25 minilm --memories M2` instead compares the
two retrievers with M2. MiniLM may download its public embedding weights on first
use, as in the existing exporter. No generator-only/hybrid results are invented;
their real providers remain separate pending handoffs.

Use `--check-only` to validate and show the planned count without writes or any
judge requests (provider readiness is not verified by this preview).
Use `--run-dir RUN_FOLDER --resume` to continue **only** the saved judging stage,
without rerunning exports or resampling conversations. The existing judge's
configuration/hash checks still apply. Use `--run-dir RUN_FOLDER --summary-only`
to display saved ratings offline. If interruption occurred before a packet
exists, preserve that partial run and start fresh instead. Failed subprocesses
stop the pipeline; previous outputs are not removed or overwritten.
