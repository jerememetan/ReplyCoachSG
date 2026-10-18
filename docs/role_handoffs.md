# Role E integration: teammate handoffs

Search the repository for **`TODO: FOR ROLE`** (VS Code: Ctrl+Shift+F).
The code markers use `# ##TODO: FOR ROLE A/C/D`; each section below explains the
exact insertion point and the checks to run. Planning notes under
`docs/superpowers/` stay ignored; this handoff guide is tracked.

## Ready now versus pending

Role E wiring is implemented: three candidate-pool modes, shared memory/ranking,
source-preserving fallbacks, mode controls, retry, demo JSON downloads and guarded
development exports. The original BM25/MiniLM comparison stays the default.

**Published:** A's 190-item train/reply bank, 50 validation queries and 60 locked
test items. **Pending:** C/D's trained-model providers; B's empirical
retrieval/memory/score calibration; the real 20-conversation hybrid proof and
persona-based LLM evaluation. Neither smoke weights nor test doubles are used by the
production provider registry. Missing models are explicitly unavailable.

## TODO: FOR ROLE A — preserve the split and support pilot checks

The reviewed split is published; preserve its manifest and privacy/audit checks.
Jereme/Role E has taken ownership of the local judging workflow, including the
runner, persona CSV, rubric, pilot and results analysis. Do not build a duplicate
runner. A retains data/split stewardship and can independently sanity-check pilot
judgments. The initial panel has **ten AI personas**, not three specialist perspectives. Each rates
every compared reply using **appropriateness, naturalness and style fit**, each
1–5, with an explanation, problem/uncertainty flags and a would-send vote. Keep personas consistent,
hide system identities and use the same roster/settings for every system.

E's `scripts/judge_replies.py` uses the standard local Gemma4 12B judge through
Ollama, without an API key or hosted calls. Keep the same judge/digest, settings,
rubric, personas and conversations across systems. These controls do not remove
judge bias; report known contextual and persona/style errors and manual spot-checks.
Use `--calibrate` for a one-persona diagnostic before a full panel; start with five validation
conversations, check privacy, judge quality and runtime. Keep correlated ratings distinct
from independent human participants, and summarise results at conversation level.
See `docs/split_and_rating_protocol.md` for the full protocol. The existing
spreadsheet scripts are optional human-review tools, not this runner.

### Published split contract

Insertion points: `src/replycoach/evaluation.py` (`DEFAULT_BANK` /
`DEFAULT_QUERIES`) and the files below. No loader-code change is needed if using
the default names; other filenames can be passed with CLI arguments.

1. Preserve the fixed 240 development / 60 test split and its duplicate-family,
   privacy, canonical-label and independent-review checks.
2. Keep the published `data/dev/train.jsonl` and `data/dev/validation.jsonl`. Both have
   `split: "dev"`; their union is the 240-item development set: 190 train and 50
   validation. Do not add a new schema split called "validation".
3. Keep the 60 locked `split: "test"` items separately at
   `data/test/test.jsonl`. Do not pass them to this development exporter.
4. Every development row must remain `completed`, explicitly privacy-safe and
   independently reviewed. IDs and normalized conversation inputs must not overlap
   between train/reply-bank and validation/query files. Group near-duplicates and
   augmentation families manually; exact-overlap guards are not a semantic detector.
5. Preserve counts by scenario/intent and split and the split seed/procedure.
   The developer export is labelled by system; anonymise and randomise outputs
   separately before giving them to AI reviewers. Do not expose ranking scores.

Readiness check (does not load models or write anything):

```powershell
.\.venv\Scripts\python.exe scripts/export_dev_outputs.py --check-only
```

The demo can use the train file as its reviewed development reply bank. Never
point it at the locked test file. Final held-out evaluation needs a separate
guarded output route coordinated by E/A; do not remove the development guard.

## Ready for Role E — evaluation output handoff

`scripts/prepare_judging_inputs.py` now adapts development exports without manual
JSON editing. It validates item/system/style coverage and fingerprints, selects
`requested_mode`, reads nested reply text, shuffles anonymous candidates and
writes `judge_inputs.jsonl` plus `PRIVATE_mapping.jsonl`. Only the public file goes
to the judges; E retains the private system mapping for analysis. Use a new
folder under ignored `experiments/judging/`. Exact fields, commands and hash/join
checks are in section 7 of `docs/split_and_rating_protocol.md`.

**TODO: FOR ROLE E with A:** a separate guarded final-test output route remains
pending. The development guard is unchanged. Preparing the packet makes no LLM
calls or ratings. E's separate `scripts/judge_replies.py` runs the persona panel
and prints averages/send percentages; see section 8 of the protocol. Its local
judge must pass pilot sanity checks before main evaluation.

## TODO: FOR ROLE C — connect the trained V4 generator

Insertion point: **`load_v4()` in `src/replycoach/generate/providers.py`**.

Return your local, task-trained V4 inference object. It must have:

- `source = "V4"`.
- `generate(message: str, context: list[str]) -> list[Candidate]`.
- Both STANDARD and LOCAL outputs from the real model, using the same style tags,
  vocabulary, token limits and preprocessing as training. The message is original
  text; context is the selected session memory, oldest to newest.
- Each Candidate: matching source, correct style, nonempty decoded string,
  `item_id=None`, finite numeric input score in `[0, 1]`, JSON-safe metadata.
  Raw negative log probabilities, constant fake confidence, unresolved decoder
  tokens, smoke models and copied saved replies are not a handoff.
- Metadata documenting model/checkpoint ID and decoding settings so exported
  results are reproducible. Explain the score calculation and its limitations.

Share the trained checkpoint, matching vocabulary/configuration, loading code and
validation outputs. Keep large weight files in ignored `artifacts/` or shared
storage; commit the provider implementation and a small experiment summary, not
the weights. Use evaluation/inference mode and bounded decoding.

## TODO: FOR ROLE D — connect the trained V5 generator

Insertion point: **`load_v5()` in `src/replycoach/generate/providers.py`**.

Use the same interface and Candidate rules as C, with `source = "V5"`. Load your
selected local mT5 base checkpoint, the trained LoRA adapter if used, and matching
tokenizer. Use the exact training input format for both styles. The helper
`src/replycoach/generate/style_control.py` is available; do not silently change
its format between training and inference.

Share model/adapter identifiers and files, tokenizer/configuration, inference
code, required local dependencies and matched validation results. No hosted
generation API. A successful GPU loading smoke test is not a trained generator.
Compare with C on the same validation set and document the selected winner.

Provider objects are cached only after a generation request. After changing model
files/configuration, restart Streamlit to clear old resources. Only one selected
generator is kept in this resource cache.

## TODO: FOR ROLE B — validate retrieval, memory and score scales

Insertion points: the experiment report you publish and the existing
`src/replycoach/rerank.py` baseline if a validated score-scaling change is agreed.

- Compare BM25/MiniLM and M0/M1/M2 on A's same validation queries, with only the
  disjoint train file indexed. Check local-term coverage on development data.
- Retrieval-only export works before any generator handoff:

```powershell
.\.venv\Scripts\python.exe scripts/export_dev_outputs.py --mode retrieval_only --retrieval bm25 --memory M2 --output experiments/retrieval_bm25_m2.jsonl
```

- Repeat with the other method/memory settings and a new output filename. Report
  reply quality and fallback counts, not ranking scores as accuracy percentages.
- The shared reranker still uses `0.6 * input_score + 0.4 * token_overlap`.
  Bounded `[0, 1]` model scores are not automatically calibrated against BM25 or
  cosine scores. Validate scales/weights with C/D before claiming a hybrid win;
  never tune them on the locked test set.

## Connected-provider verification

Run the tests, then use the real model and development files:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_generators.py tests/test_systems.py tests/test_evaluation.py -q --basetemp .pytest-basetemp-handoff-new
.\.venv\Scripts\python.exe scripts/export_dev_outputs.py --generator v4 --mode all --output experiments/dev_three_modes_v4.jsonl
```

The provider-pending tests must be updated when a real implementation replaces a
pending factory; preserve missing-artifact/contract tests by injecting loader
failures rather than depending on a teammate's local weights.

`all` calls each source once per query and reuses the candidates across the three
mode pools. The exporter refuses incomplete comparisons, unsafe/unreviewed data,
held-out rows, input overlap, dataset output paths and existing output files. It
does not append to the runtime log. Exports are UTF-8 JSONL with explicit modes,
origins, counts, selected memory and dataset fingerprints, not blinded surveys or
automatic evidence of contextual correctness.
