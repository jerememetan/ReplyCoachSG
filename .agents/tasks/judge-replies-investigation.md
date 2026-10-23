# `judge_replies.py` investigation

## Summary answer

`scripts/judge_replies.py` is an offline-evaluation runner for anonymous Reply Coach development outputs. Given a prepared packet folder, it validates the public candidates and private system mapping, loads a panel of fictional student personas and a shared rating rubric, then asks a locally running Ollama model to score every candidate/persona pair. It records three 1–5 scores, a simulated `would_send` vote, a short explanation, a problem category, and an uncertainty flag.

It does not generate Reply Coach answers, alter the dataset, create training labels, or contact a hosted LLM. Normal judging makes only local HTTP requests to the loopback Ollama server (`GET /api/tags`, then `POST /api/chat`); `--check-only`, `--summary-only`, and `--help` do not make model requests. Results are written under `PACKET/ratings/` and should be treated as simulated, provisional evaluation evidence rather than human preference or accuracy.

## Evidence and behavior

### Command-line interface

`main()` in `scripts/judge_replies.py` defines this interface, confirmed with the safe command `\.venv\Scripts\python.exe scripts/judge_replies.py --help`:

- Required positional `packet`: a prepared development packet directory.
- `--personas PATH`: persona CSV; default `data/evaluation/personas.csv` through `replycoach.io.ROOT`.
- `--env-file PATH`: simple environment file; default `.env` at the repository root.
- `--out-dir PATH`: alternate results directory; otherwise `PACKET/ratings`.
- `--resume`: continue a previously created run, skipping successful judgments and retrying failures.
- `--check-only`: validate packet/personas/configuration without Ollama requests, ratings, or output files.
- `--summary-only`: read an existing saved run and display its summary without Ollama requests.
- `--check-only` and `--summary-only` are mutually exclusive.

The parser description is “Rate anonymous development replies with local Ollama; no agent framework.”

### Input files and formats

1. **Packet folder** — `load_packet()` in `scripts/judge_replies.py` requires both UTF-8 JSONL files:
   - `judge_inputs.jsonl`: a nonempty public packet.
   - `PRIVATE_mapping.jsonl`: one private mapping row for every public candidate.

   `replycoach.io.read_jsonl()` in `src/replycoach/io.py` parses each nonblank line as JSON and reports malformed-line numbers. The public row must contain exactly these fields: `schema_version`, `candidate_id`, `conversation_id`, `conversation`, `incoming_message`, `style`, and `reply`. The supported schema version is integer `1`; candidate IDs must be unique and present exactly once in the private mapping; styles must be `STANDARD` or `LOCAL`; conversation IDs and replies must be nonempty strings; and conversation turns must be `{speaker, text}` objects with speakers `user` or `peer`. The incoming message must be from `peer`.

   The private mapping is checked against the SHA-256 digest of the public file, has unique nonempty candidate IDs, and must agree on each candidate’s conversation ID and style. Its required metadata values include nonempty `system`, `memory_mode`, `retrieval_method`, and `requested_mode`. The loader also verifies that every system covers the same conversation/style cells, preventing unmatched system comparisons. It does not send private rows to Ollama.

2. **Persona CSV** — `load_personas()` requires exactly the columns `id,description`, nonempty values, and unique IDs. The repository default is `data/evaluation/personas.csv`, which currently contains ten fictional profiles (`P01`–`P10`). Persona text supplies perspective and habits, but the shared rubric remains the scoring standard.

3. **Rubric** — `RUBRIC_PATH` points to `scripts/data-split/rating_rubric.md`. It defines the anchors for appropriateness, naturalness, style fit, `would_send`, problem categories, and uncertainty. The rubric explicitly says to rate the reply rather than the author, not rewrite replies, and not create gold labels.

4. **Environment file** — `read_env()` reads only simple `KEY=value` lines and only recognizes `OLLAMA_MODEL` and `OLLAMA_BASE_URL`; it does not execute or expand `.env` contents. Process environment variables override values from the file. Defaults are `OLLAMA_BASE_URL=http://127.0.0.1:11434` and `OLLAMA_MODEL=qwen2.5-coder:7b`.

The packet producer, `scripts/prepare_judging_inputs.py`, calls `replycoach.judging.prepare_judging_packet()` and writes the two packet files without model calls. Its output contract and privacy boundary are documented in `docs/split_and_rating_protocol.md`: evaluators should receive only `judge_inputs.jsonl`, while the private mapping stays with the local analyst.

### Processing flow

`main()` performs the following sequence:

1. Parse arguments and call `load_packet()` before any judging call.
2. Resolve the output path and refuse paths equal to, or inside, the repository `data/` directory.
3. For ordinary runs, load personas, read the environment configuration, validate that the Ollama URL is HTTP loopback-only, reject empty/cloud model names, and calculate `len(public candidates) × len(personas)` expected judgments.
4. Unless `--check-only` is selected, call Ollama `GET /api/tags` and require an installed local model with a digest. It accepts the exact requested model name or its `:latest` form, and rejects entries indicating a remote host/model. It never pulls a model.
5. Read the rubric and build a frozen `run.json` manifest containing packet hashes, persona data and hash, rubric, script hash, response schema, deterministic generation options, Ollama base URL, model name, and installed model digest. A hash of that manifest becomes `run_id`.
6. Acquire `PACKET/ratings/.lock` using `filelock.FileLock` with a zero-second timeout, then validate any existing `ratings.jsonl` rows against this run. Successful duplicate rows or mismatched IDs/configuration are rejected.
7. Iterate candidates in public-file order and personas in CSV order. Existing successful `(candidate_id, persona_id)` pairs are skipped; failed pairs are retried on `--resume`.
8. After each candidate, print a refreshed summary. At the end, write `summary.csv` and return success only when all expected pairs have valid ratings.

### Model request and judgment logic

`judge_one()` deliberately creates a fresh metadata-free context for each judgment. The system prompt contains:

- a fictional-persona instruction and the selected persona description;
- the complete shared rubric;
- the rule that `would_send=true` means the persona would send the reply with little or no editing;
- an instruction that quoted conversation is untrusted data, not instructions;
- an instruction not to rewrite the reply; and
- the required JSON schema.

The user message contains only the candidate’s `conversation`, `incoming_message`, `style`, and `reply`. It excludes candidate IDs, system labels, private mapping fields, retrieval scores, and provenance, preserving system blinding in the model prompt.

The request is sent to `POST /api/chat` with `stream=false`, the selected local model, Ollama structured-output `format=SCHEMA`, and `OPTIONS = {temperature: 0, seed: 425, num_predict: 384, num_ctx: 8192}`. `request_json()` disables HTTP proxies and `NoRedirect` refuses redirects, so task text cannot be redirected to another host. The request timeout is 180 seconds.

A response is accepted only when it is a JSON object with `done=true`, a non-length `done_reason`, a message object, and content that parses as JSON. `validate_rating()` requires exactly these fields and no extras:

```json
{
  "appropriateness": 1,
  "naturalness": 1,
  "style_fit": 1,
  "would_send": false,
  "explanation": "A short reason.",
  "problem": "none",
  "uncertain": false
}
```

The three scores must be actual integers from 1 through 5. `would_send` and `uncertain` must be JSON booleans. The explanation must be a nonblank string of at most 500 characters. `problem` must be one of `none`, `garbled_or_unreadable`, `offensive_or_unsafe`, `personal_info`, or `generic_or_unrelated_fallback`.

Each pair gets at most two attempts in one run (the initial attempt plus one retry). Request errors, malformed JSON, incomplete Ollama responses, and validation errors become an `error` row with no rating; no score is fabricated. The exact catches are `OSError`, `URLError`, `ValueError`, and `TypeError`.

### Scoring, aggregation, and displayed results

`SCORES` contains `appropriateness`, `naturalness`, and `style_fit`; there is no combined score and no automatic winner. `summarise()`:

- retains only `status == "ok"` ratings for score calculations;
- counts failed rows separately;
- averages each candidate’s successful persona ratings, including the boolean `would_send` as a percentage;
- averages those candidate-level values within each `(system, memory_mode, retrieval_method, requested_mode, style)` group; and
- reports `rated`, `expected`, `failed`, and the number of populated candidate/conversation cells.

`show_summary()` prints one row per system setting/style with two-decimal averages and a `Rated/Expected` count. It labels `Send %` as simulated willingness, not human approval, and marks incomplete rows as provisional. The protocol in `docs/split_and_rating_protocol.md` gives the same operational warning: failures are missing rather than zero votes, and incomplete matched coverage must not be used to declare a winner.

The private mapping is used locally by `summarise()` to attach system and configuration labels to anonymous candidate IDs. Because the packet contract supplies one candidate per system/conversation/style cell, the candidate-level averaging corresponds to averaging persona ratings within each cell and then averaging cells equally. The script itself does not compare different systems inside the model call.

### Outputs and side effects

For a normal run, `main()` creates or reuses an output directory outside `data/` and writes:

- `run.json`: frozen run manifest and configuration fingerprints;
- `ratings.jsonl`: append-only one-row-per-candidate/persona journal, including status, rating or error, attempts, timestamp, Ollama timing counters when available, and elapsed seconds;
- `summary.csv`: regenerated aggregate report; and
- `.lock`: a lock file used to prevent simultaneous writers.

The journal is flushed and `os.fsync()` is called after every judgment, so completed rows survive most interruptions. `KeyboardInterrupt` exits with code 130 and explains that completed judgments are saved; `--resume` can continue. A malformed final JSONL line from an abrupt disk/process interruption is not automatically truncated and requires manual inspection/recovery, as documented in the protocol.

Exit behavior is:

- `0` for a complete matched run or complete summary-only result;
- `1` for an incomplete run or summary with missing/failed judgments; and
- `2` for validation, configuration, I/O, lock, or other handled operational errors. `--help` exits normally without loading a packet or contacting Ollama.

A fresh run uses `out.mkdir(..., exist_ok=args.resume)` and creates `run.json` with exclusive creation, so an existing output directory/run is intentionally not silently overwritten. `--resume` requires the saved manifest to match exactly; changed packet, personas, rubric, script, model/configuration, or options require a fresh output directory. `--summary-only` uses the saved manifest’s personas/configuration and checks that the current packet hashes still match before reading saved ratings.

### Security and privacy boundaries

The loopback URL restriction, disabled proxy, disabled redirects, model-installed check, fresh metadata-free prompt, and private mapping hash protect against common accidental disclosure/configuration mistakes. The script still sends the public conversation and reply text to the local Ollama process; public packet text is not automatically redacted for personal information. `PRIVATE_mapping.jsonl` is protected by project convention and `.gitignore` location, not filesystem access control, and custom output directories are not automatically ignored.

The repository `.gitignore` ignores `experiments/judging/` and `.env`, but the script allows a custom `--out-dir`; users must keep custom packet/results locations private themselves. The current working tree already had unrelated/pre-existing modifications and untracked judging files when inspected; this investigation did not change them.

## Conclusions and recommendations

- Use `--check-only` first. It verifies the packet, persona CSV, URL shape, and model-name rules without contacting Ollama or creating ratings.
- Keep the public packet and private mapping together locally for analysis, but give an evaluator/model only `judge_inputs.jsonl`. Inspect replies for privacy leakage because anonymization does not redact text.
- Use a fresh output directory whenever the packet, personas, rubric, script, model, or generation options change. Use `--resume` only for the exact frozen run, and use `--summary-only` for offline display.
- Do not rank systems from an exit-code-1/incomplete run or treat `would_send` as human approval. Pilot-check the provisional Qwen2.5-Coder evaluator, especially for Singlish/Mandarin judgments, before treating results as experiment evidence.
- The main operational limitation is that results are correlated persona-model simulations, not independent human ratings. If the project needs human-validity claims, add a separate human sanity-check/calibration protocol rather than interpreting these scores as population statistics.
- For robustness, future maintenance could add an explicit privacy scan/redaction review for public packet text and a recovery tool for interrupted malformed `ratings.jsonl` tails; neither is currently performed automatically.

## Verification performed

- Read `scripts/judge_replies.py` and directly referenced `src/replycoach/io.py`, `scripts/data-split/rating_rubric.md`, and `data/evaluation/personas.csv`.
- Read the packet producer and relevant protocol/architecture documentation: `scripts/prepare_judging_inputs.py`, `docs/split_and_rating_protocol.md`, `docs/architecture.md`, `roles/E_rerank_demo/START_HERE.md`, `.gitignore`, `pyproject.toml`, and `README.md`.
- Ran the safe help check with the repository virtual-environment interpreter. It printed the expected usage/options and made no Ollama request.
- No source files, configuration, servers, model calls, commits, or repository data were modified by this investigation.
