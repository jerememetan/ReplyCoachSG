# Four-system validation comparison — 26 October 2026

The main ten-persona evaluation **completed at approximately 15:26 SGT on
26 October**: 4,000/4,000 successful judgments, zero recorded errors, and 500/500
ratings in every system/style group. The runner reported a complete matched run;
its offline summary check also passed. No evaluation process remains active.

## Completed validation summary

| System (M2) | STANDARD appropriateness / 5 | LOCAL appropriateness / 5 | STANDARD would-send | LOCAL would-send |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 2.678 | 3.476 | 29.0% | 53.8% |
| MiniLM | 2.976 | 3.802 | 40.0% | 65.0% |
| V4 | 2.408 | 2.710 | 23.6% | 33.8% |
| V5 | 3.666 | 4.110 | 61.2% | 76.8% |

V5 has the highest average appropriateness and simulated would-send percentage
in both styles in this run. These are persona-based AI judgments on 50 validation
conversations, not human approval or held-out test results. Detailed paired
comparisons, uncertainty intervals and ten qualitative cases are now in the
[full comparison report](reports/four_system_comparison_20261026.md).
Independent teammate sanity checks remain. A later receiver check reproduced
6/6 V5 reply texts on a fixed three-conversation subset; see
[V5 local verification](../roles/D_transformer/LOCAL_VERIFICATION.md).
Naturalness, style-fit and complete coverage counts are saved in
`experiments/judging/four_systems_20261026_01/main_packet/ratings/summary.csv`.
The methodological limitations recorded below still apply.

## Scope and provenance

All systems answer the same 50 published validation conversations in both
STANDARD and LOCAL styles. All exports use M2. Retrieval uses the separate
190-item training/reply bank. No held-out test examples are read or evaluated.

| Label | Selected mode | Saved output file |
| --- | --- | --- |
| BM25_M2 | Retrieval only | `experiments/retrieval_bm25_m2.jsonl` |
| MiniLM_M2 | Retrieval only | `experiments/retrieval_minilm_m2.jsonl` |
| V4_M2 | Generator only | `artifacts/role_c_runs/20261025_received_handoff/receiver_three_modes_v4.jsonl` |
| V5_M2 | Generator only | `ROLE_4 HANDOFF/v5_ul025_M2.jsonl` |

The existing preparation tool verified complete item/style coverage, matching
dataset fingerprints and incoming messages, internally consistent run settings,
and no recorded errors or degraded modes. Eight source/configuration files were
hashed before the run and checked unchanged after the diagnostic. The receipt
records these hashes; the private packet mapping also hashes each source export.

V4's inference and saved outputs were previously reproduced locally. Its 100
selected replies include **72 generated replies and 28 genuine safety
fallbacks**; all are retained. V5's 100 replies are teammate-supplied saved
generations. Subsequent local verification reproduced six saved reply texts, and
the 27 October live demo ran all four sources successfully; see the
[ranking work log](all_models_ranking_worklog.md). These follow-up checks do not
independently reproduce all 100 V5 outputs or its training history. The original
evaluation still uses the supplied export. The generator exports' `retrieval_method=bm25` field
is configuration metadata: both have zero retrieved candidates.

## Anonymous packets

Run folder: `experiments/judging/four_systems_20261026_01/` (ignored by Git).

- `pilot_packet/`: five conversations, four systems, two styles = 40 candidates.
- `main_packet/`: 50 conversations, four systems, two styles = 400 candidates.
- Sampling/shuffle seed: 426. Every system uses the same selected conversations.
- Only `judge_inputs.jsonl` fields go to the judge: conversation, incoming peer
  message, requested style and exact candidate reply. The local anonymous IDs
  are not included in the model prompt. Original speaker labels are retained.
- `PRIVATE_mapping.jsonl` stays local. System names, scores, authored reference
  replies, checkpoint metadata and pre-judge expectations are excluded from prompts.

The seed provides a reproducible sample, not a claim that these examples have
never appeared in any earlier diagnostic. There was no output-based sample
selection or prompt revision in this run.

## Completed one-persona diagnostic

The existing `--calibrate` mode used P01 unchanged. All **40/40 unique
candidate/persona pairs** succeeded, with no error rows or uncertainty flags.
This is diagnostic coverage, not a measured correctness rate or a ten-persona pilot.

Judge: local `gemma4:12b`; digest
`6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b`.
Temperature 0, seed 425, thinking off, context 4096, output budget 2048.
The existing rubric/personas and runner were unchanged and are recorded in `run.json`.

Before requesting ratings, the assistant recorded 12 good/bad expectations in
`PRE_JUDGE_SPOTCHECK.md`. The judge rejected six of seven assistant-flagged
unsuitable replies and accepted all five plausible controls by would-send.
These are **assistant inspections, not independent human gold labels or an
estimate of judge accuracy**. They do not establish that all 40 ratings are correct.

| Case | Reply | Observed judgment |
| --- | --- | --- |
| Peer needs a new place after a booking expires | BM25 LOCAL: "ok, need email prof?" | Appropriateness 4, send yes. A miss: the explanation invents a relevant role for a professor. |
| Peer asks about Gaussian elimination | BM25 LOCAL: "uhh, i think still need do slides" | Appropriateness 2, send no; correctly identifies the topic shift. |
| Same maths question | V4 LOCAL: "ok i will be there tonight" | Appropriateness 1, send no; correctly rejects an unrelated arrival message. |
| Saturday availability and a night-market invitation | V5 STANDARD: "Sure, I'll be free on Friday." | Appropriateness 1, send no; flags the unsupported day. |
| Peer proposes a Thursday make-up class | MiniLM STANDARD: "yeah, we don't have class on tuesday" | Appropriateness 2, send no; correctly identifies the missed availability question. |
| Peer invites the user to run | BM25 STANDARD: "sure. we can find a quiet place in the library" | Appropriateness 1, send no; correctly rejects the wrong activity. |
| Peer offers a night-market invitation | MiniLM STANDARD: "Sure! What time should we meet?" | Appropriateness 5, send yes; plausible acceptance and follow-up. |

Two additional limits matter for interpretation:

1. **Score variation:** C000025 and C000029 have identical conversation, requested
   style and reply ("I think a reservation is required, hold on let me check").
   Under the same persona/settings, appropriateness was 2 versus 3. Both were
   rejected by would-send. Fixed settings do not guarantee identical judgments.
2. **Style ceiling:** 39/40 style-fit scores were 5. The sample may contain many
   acceptable styles, but this does not demonstrate sensitive style discrimination.

The raw diagnostic averages are in `pilot_packet/ratings/summary.csv`. With only
five conversations and one persona, they do not select a winning generator or
establish a general improvement. No ratings were rewritten to match inspection.

## Main evaluation setup and completion

The completed main job used the existing runner on all 400 candidates with all ten original
personas: **4,000 successful judgments**. It used the unchanged judge, rubric and
settings; diagnostic ratings are not pooled into this run. This is a preliminary
AI comparison with the above limitations, not a declaration that the judge is
validated or equivalent to human reviewers. Independent teammate sanity checks
remain outstanding. Ten personas share one model and are correlated.

The diagnostic's warm median was 14.092 seconds per judgment, suggesting about
15.7 hours for 4,000 judgments before additional overhead/retries. This is a rough
runtime estimate, not a completion time. The diagnostic summed 657.362 seconds
of recorded judgment durations, including its initial model load.

Progress is saved after each judgment in `main_packet/ratings/ratings.jsonl`.
`main_console.txt` contains progress and provisional summaries. The runner writes
`main_packet/ratings/summary.csv` when it finishes; a file's existence alone does
not prove complete coverage. Check for 4,000 unique successful pairs and the
runner's complete-run status before interpreting the final averages.

Display saved progress without model requests:

```powershell
.\.venv\Scripts\python.exe scripts/judge_replies.py experiments/judging/four_systems_20261026_01/main_packet --summary-only
```

An incomplete run deliberately returns a nonzero status. After an interrupted
process has stopped, continue without rerating completed pairs:

```powershell
$env:OLLAMA_MODEL = 'gemma4:12b'
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
.\.venv\Scripts\python.exe scripts/judge_replies.py experiments/judging/four_systems_20261026_01/main_packet --resume
```

After completion, analyse paired results by conversation and style, retain
fallback coverage and judge-error examples, and distinguish simulated willingness
to send from human approval. Do not claim a ranking from partial averages or
use AI ratings as training labels. No app, dataset or model changes were needed.
