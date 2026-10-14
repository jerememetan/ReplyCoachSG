# Role E integration: teammate handoffs

Search the repository for **`TODO: FOR ROLE`** (VS Code: Ctrl+Shift+F).
The code markers use `# ##TODO: FOR ROLE A/C/D`; each section below explains the
exact insertion point and the checks to run. Planning notes under
`docs/superpowers/` stay ignored; this handoff guide is tracked.

## Ready now versus pending

Role E wiring is implemented: three candidate-pool modes, shared memory/ranking,
source-preserving fallbacks, mode controls, retry, demo JSON downloads and guarded
development exports. The original BM25/MiniLM comparison stays the default.

**Pending:** A's official data split; C/D's trained-model providers; B's empirical
retrieval/memory/score calibration; the real 20-conversation hybrid proof and
official blind evaluation. Neither smoke weights nor test doubles are used by the
production provider registry. Missing models are explicitly unavailable.

## TODO: FOR ROLE A — publish the reviewed split

Insertion points: `src/replycoach/evaluation.py` (`DEFAULT_BANK` /
`DEFAULT_QUERIES`) and the files below. No loader-code change is needed if using
the default names; other filenames can be passed with CLI arguments.

1. Freeze the fixed 240 development / 60 test split after checking duplicate
   families, privacy, canonical labels and the existing independent reviews.
2. Publish `data/dev/train.jsonl` and `data/dev/validation.jsonl`. Both must have
   `split: "dev"`; their union is the 240-item development set. Agree the internal
   validation size with B/C/D; do not add a new schema split called "validation".
3. Publish the 60 locked `split: "test"` items separately, e.g.
   `data/test/test.jsonl`. Do not pass them to this development exporter.
4. Every development row must remain `completed`, explicitly privacy-safe and
   independently reviewed. IDs and normalized conversation inputs must not overlap
   between train/reply-bank and validation/query files. Group near-duplicates and
   augmentation families manually; exact-overlap guards are not a semantic detector.
5. Publish counts by scenario/intent and split, the split seed/procedure, and your
   blind-rating protocol. The developer export is labelled by system; anonymise
   and randomise outputs separately before giving them to raters.

Readiness check (does not load models or write anything):

```powershell
.\.venv\Scripts\python.exe scripts/export_dev_outputs.py --check-only
```

The demo can use the train file as its reviewed development reply bank. Never
point it at the locked test file. Final held-out evaluation needs A's separate
frozen protocol/runner; do not remove the development guard to obtain test outputs.

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
