# V4 handoff receipt and local verification — 25 October 2026

The supplied V4 inference handoff is installed and runnable in the receiving
checkout. This records actual receiver-side checks, not another training run or
a claim that reply quality is good.

## Inventory and placement

All **28 manifest-listed files** passed SHA-256 checks. The bundle has 30 files
including its manifest and bundle readme. It contains no linked/reparse entries.

| Deliverable | Repository location | Action |
| --- | --- | --- |
| Task-trained V4 checkpoint | `artifacts/rnn_v4_task.pt` | Installed; matches the handoff SHA-256 below. |
| Training, inference and verification code | Existing `src/replycoach/generate/` and `scripts/` files | Retained existing copies; all supplied content matches, allowing Windows CRLF differences. |
| Handoff, experiment report, role guide and licence checklist | Existing `roles/C_rnn_attention/` and `docs/licence_checklist.md` | Retained the existing reports; linked this receiver verification from the handoff. |
| 13 raw evidence files and two bundle metadata files | `artifacts/role_c_runs/20261025_received_handoff/` | Copied and hash-verified; includes V3/V4 histories, validation generations, comparison, audit, environment and sender verification logs. |
| Original received package | `role_c_v4_handoff/` (removed) | Rechecked installed copies, then moved the original folder to the Recycle Bin at the user's request. Recoverable, but no longer in the workspace. |

Fourteen source/document files were already present; no production source or
training code was overwritten. Sixteen missing private files were installed.
The checkpoint and dated raw evidence are already covered by artifact ignore rules.

Checkpoint SHA-256:

```text
a8b4f0e1f554b86926959ce0a5df4c7a297bdaa71fe03ecf91e67afcb93d900f
```

## Checks executed on the receiving machine

- `load_v4()` loads the default installed model as **V4**, epoch **8**, with its
  embedded **5,251-token vocabulary** and both STANDARD/LOCAL style tokens.
- Its recorded train/validation fingerprints exactly match the current 190-item
  train bank and 50 validation items. The datasets were not rewritten.
- Recomputed V4 task-validation loss: **4.1619403949**, versus the supplied
  **4.1619404684**; difference is within the verifier's `1e-5` tolerance.
- Regenerated all **100 V4 validation replies**. Every decoded text matched;
  scores matched within `1e-6`. References and token-F1 diagnostics also matched.
- Checked all **100 supplied V3 evidence rows** for item/style coverage,
  references and token-F1 consistency. V3 weights were **not** reloaded.
- Re-exported all three modes on the same 50 queries with BM25/M2. All **150
  fresh mode records** were complete and reproduced the supplied selected replies,
  origins, pool counts and scores within `1e-6`.
- Exercised the real Streamlit app using the reviewed train bank: requested and
  effective modes were **Hybrid**, with **6 retrieved / 2 generated candidates**
  and no missing-model warning. Reply selection filled only the display composer;
  conversation input remained unchanged.
- Full software suite with installed weights: **254 passed in 97.82 seconds**.
  Test cache was disabled and pytest temporary files were outside the repository.

Fresh outputs and machine-readable results remain private/ignored in the dated
receipt folder: `receiver_three_modes_v4.jsonl`, `receiver_verification.json` and
`receiver_demo_log.jsonl`. The isolated UI run's new log was archived there;
existing runtime-history state was preserved. UI screenshot:
`docs/superpowers/screenshots/2026-10-25-v4-receiver-hybrid.jpg` (ignored).

## Reproduced selection counts and quality limitations

| Mode | Selected retrieval replies / 100 | Selected V4 replies / 100 | Fallbacks / 100 |
| --- | ---: | ---: | ---: |
| Retrieval only | 100 | 0 | 0 |
| Generator only | 0 | 72 | 28 |
| Hybrid | 100 | 0 | 0 |

These checks establish a functioning and reproducible inference handoff, not a
hybrid quality improvement. All 100 final hybrid replies still came from
retrieval. Reference-overlap diagnostics and model likelihoods are not measures
of contextual correctness or style quality. Keep the limitations in
`EXPERIMENT_REPORT.md` and the judge pilot report when interpreting these results.

## What this bundle does not establish

This is an **inference integration bundle**, not a complete full-training archive.
V3 task weights, original V3/V4 public checkpoints/reports and DailyDialog inputs
are separate resources, as stated in the bundle readme. Their absence from this
bundle does not prevent running V4, but the complete all-model
`scripts/verify_role_c.py` cannot be rerun from this bundle alone. No retraining
or independent reproduction of historical public validation loss was performed.

V5, matched V4/V5 quality judging, score calibration, the final 20-conversation
demo proof and final held-out evaluation remain separate work. Frozen data and
production source were preserved. The locked test file was only integrity-hashed;
its examples were not inspected or used for model inputs, training or tuning.

## Use the installed model

Restart an existing Streamlit server after installing/changing model weights:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/replycoach/demo_app.py
```

Select **System settings → Hybrid → V4 (attention-RNN)** and **Reply bank settings
→ Reviewed development file**, with `data/dev/train.jsonl` as the path.

For another matched export, choose an unused output filename; for example:

```powershell
.\.venv\Scripts\python.exe scripts/export_dev_outputs.py --generator v4 --mode all --retrieval bm25 --memory M2 --output experiments/v4_received_verify_01.jsonl
```

No commit or push was performed as part of receiving this bundle.
