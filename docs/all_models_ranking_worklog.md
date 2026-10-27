# All-model ranking work log

Later follow-up: the separate [fine-tuned V6 experiment](../roles/E_rerank_demo/V6_TRAINING.md)
completed on 27 October. This log records the original frozen-scorer baseline;
the trained option and its comparisons are documented in that handoff.

Design and implementation: 26-27 October 2026, Asia/Singapore.
The user approved `all_models_ranking_design.md` before implementation.

## What can be used now

In the existing demo, open **System settings**, choose **All models
(experimental)** and click **Suggest two replies**. BM25, MiniLM, V4 and V5 each
propose Standard and Local replies. A shared local scorer selects one per style.
The two selected replies fill the display-only composer when clicked. The model
proposal table retains all eight outputs, their origins and selected markers.

The default remains **Retrieval comparison**. A source failure is explicitly
labelled as a partial comparison; a ranker failure retains proposals and offers
retry without silently replacing the selector with the old heuristic.

## Implementation record

| Work item | Status / evidence |
| --- | --- |
| V5 inference dependencies | Installed PEFT 0.21.2, SentencePiece 0.2.2 and Accelerate 1.15.0 in the project environment; updated the `mt5` extra's PEFT minimum |
| Real V5 verification | 6/6 exact saved reply texts across three validation conversations; see `roles/D_transformer/LOCAL_VERIFICATION.md` |
| Shared scorer | `src/replycoach/shared_ranker.py`: pinned CPU cross-encoder, separate style pools, exact whitespace-normalized deduplication, all contributing origins retained |
| Four-source collection | `src/replycoach/all_models.py`: existing per-source displayed proposals and fallbacks, isolated failures, speaker alignment and timing metadata |
| Demo | Separate all-model renderer, two cached generators and one cached scorer; existing modes/exports preserved |
| Replay | `scripts/replay_shared_ranker.py` validates the development packet, source hashes, complete paired ratings and manifest before loading a model |
| Setup and licence | README setup/replay commands and model licence inventory updated |

The scorer is `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, on CPU. It receives conversation,
incoming message and candidate reply only. Model identity, source scores and
ratings do not enter its query. Raw logits are ranking signals; the legacy
fallback threshold is not applied to them. Context truncation is recorded.

## Saved-proposal validation result

**The frozen shared scorer did not outperform always choosing V5. Keep this mode
experimental.** Its point estimates improve over the legacy mixed-score selector,
but appropriateness intervals against that selector include zero. There is no
clear advantage over fixed MiniLM. No subsequent rule tuning was performed.

This replay uses the exact 400 saved proposals for 50 validation conversations,
and the existing 4,000 successful AI-persona ratings. It makes **zero new judge
calls**, generates no new replies and trains no model. All policies select from
the same four source proposals per style; the legacy comparator uses their
already-reranked original scores, without applying its formula twice.

Appropriateness, naturalness and style fit are mean 1-5 ratings. Would-send is a
simulated persona response rate, not human survey accuracy.

| Selector | Style | Appropriate | Natural | Style fit | Would-send |
| --- | --- | ---: | ---: | ---: | ---: |
| Shared cross-encoder | Standard | 3.060 | 4.916 | 5.000 | 43.8% |
| Shared cross-encoder | Local | 3.796 | 4.756 | 4.892 | 67.4% |
| Always V5 | Standard | 3.666 | 4.964 | 5.000 | 61.2% |
| Always V5 | Local | 4.110 | 4.956 | 4.998 | 76.8% |
| Always MiniLM | Standard | 2.976 | 4.896 | 5.000 | 40.0% |
| Always MiniLM | Local | 3.802 | 4.846 | 4.954 | 65.0% |
| Legacy mixed-score selector | Standard | 2.678 | 4.862 | 5.000 | 29.0% |
| Legacy mixed-score selector | Local | 3.476 | 4.658 | 4.862 | 53.8% |

Paired appropriateness difference (shared minus comparator), with descriptive
95% conversation-bootstrap intervals, 20,000 resamples, seed 425:

| Comparator | Standard | Local |
| --- | --- | --- |
| Always V5 | -0.606 [-1.076, -0.160] | -0.314 [-0.672, 0.038] |
| Always MiniLM | +0.084 [-0.396, 0.568] | -0.006 [-0.374, 0.346] |
| Legacy | +0.382 [-0.140, 0.908] | +0.320 [-0.116, 0.744] |

Shared selection's would-send difference versus V5 is -17.4 percentage points
for Standard (interval -31.0 to -4.2) and -9.4 for Local (-20.6 to +1.6).

Canonical selected source counts (50 per style):

| Source | Standard | Local |
| --- | ---: | ---: |
| BM25 | 15 | 10 |
| MiniLM | 13 | 20 |
| V4 | 5 | 1 |
| V5 | 17 | 19 |

Duplicates retain every contributor. For joining frozen ratings only, the lowest
anonymous candidate ID is used; ratings are never cherry-picked. Counts above
refer to that canonical record, so they differ from contributor counts. No
safety fallback was selected in this replay.

Private reproducibility evidence is under
`experiments/judging/four_systems_20261026_01/shared_ranker_replay_20261026_01/`:
`result.json` includes candidate IDs, all differences, scorer identity and input
hashes; `summary.csv` contains the table aggregates. README provides the commands.
The check-only command validated 50 conversations, 400 proposals and 4,000 ratings
before model loading. CPU scorer loading took 19.00 seconds; scoring all 50
conversations took 3.47 seconds after loading. These are one-run measurements.

## Verification and limits

The full test suite passed **350 tests in 100.05 seconds** after implementation,
and **350 in 131.42 seconds** when rechecked on 27 October after the app reset.
Tests cover source-score independence, deterministic duplicates/ties, separate
style selection, repeated/consecutive speakers, invalid scores, missing sources,
ranker failure, retry, click-to-fill, stale result clearing and replay integrity.
The missing-V5 UI fixture explicitly points to a nonexistent test checkpoint so
unit tests never accidentally load the locally installed real model.

The real browser check on 27 October completed **4/4 models** against the
190-example bank. All eight source proposals appeared, including a correctly
labelled V4 Local safety fallback. The selected Standard reply came from V5;
Local had identical BM25/MiniLM contributors. Clicking Standard filled the
display-only composer while preserving both input fields. Screenshot proof is
saved locally at `docs/superpowers/screenshots/2026-10-27-all-models.png` (ignored).
The complete live result is logged as `all_models_suggestions` in the ignored
`experiments/logs/runs.jsonl`, with no source or ranking error.

This first live request measured BM25 0.48 s, MiniLM 74.28 s, V4 0.18 s,
V5 18.76 s and shared ranking 1.11 s. These include model initialization and
network/cache effects, so they are not warm latency estimates. Streamlit's
source watcher subsequently probed unused optional Transformers modules and
printed warnings. The README launch command disables that watcher with
`--server.fileWatcherType none`; source edits then require restarting the app.

Final integrity checks confirmed all eight original input/export/rubric hashes
and the 4,000-rating file hash were unchanged. Rating SHA-256:
`3feea90473ef69e630f43ea92fa4c2cb2061d6e1fceae9187010d01403634d81`.
The replay check-only command was rerun successfully on 27 October with zero
model calls.

Full test command (two CPU threads, fresh system-temp directory, cache disabled):

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'
$rankTestTemp=Join-Path $env:TEMP ('replycoach-all-models-'+[guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp $rankTestTemp
```

The V5 three-conversation check measured a 50.92-second cold load and
4.51-6.44 seconds per warm reply pair on CPU-only PyTorch. First all-model use also
loads MiniLM and the scorer; these figures do not promise full-request latency.

The cross-encoder was trained for passage relevance, not conversational Singlish
reply quality. Frozen AI persona judgments remain correlated evaluations from
one judge; the bootstrap conditions on those outputs and is not a human-user
confidence interval. This is counterfactual selection on saved proposals, not a
new live-generation quality experiment.

Explicit speakers are preserved for the shared scorer. The existing V5 provider
still guesses alternating speakers from its plain-string context; changing that
source interface would require separate evaluation. The replay uses M2 only;
live M0/M1 are available without separate quality claims.

The 190/50 development split, 60 locked test items, model weights, original
exports and completed ratings are preserved. No new training, live Gemma judging,
test-set inspection, PR or push is included in this implementation.
