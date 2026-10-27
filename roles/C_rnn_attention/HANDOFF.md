# Role C handoff — 20 October 2026

V4 is connected through `load_v4()` in `src/replycoach/generate/providers.py`.
This is a working experimental generator with substantial quality limitations;
it has not been selected over V5 and is not evidence of a hybrid quality gain.

**Receiver update, 25 October:** the supplied checkpoint is installed locally;
100 V4 validation generations and all 150 three-mode records were reproduced.
See [LOCAL_VERIFICATION.md](LOCAL_VERIFICATION.md) for inventory, installed paths,
actual checks and the distinction from full training reproduction.

## Artifacts

- Default deployed task checkpoint: `artifacts/rnn_v4_task.pt`.
- Complete measured run: `artifacts/role_c_runs/20261020_colab_handoff/`.
- Matching vocabulary, dimensions, token limits, selected epoch, training/validation
  IDs and data fingerprints are embedded in the checkpoint. No separate tokenizer
  download is needed. The checkpoint format is `replycoach-rnn-task-v1`.
- Task V4 SHA-256: `a8b4f0e1f554b86926959ce0a5df4c7a297bdaa71fe03ecf91e67afcb93d900f`.
- Original five-epoch Colab V4 and report were recovered from the private
  [Drive results folder](https://drive.google.com/drive/folders/1swSOfJibtGp-iw_1ejiOWCsDZVVNEA0I).
  Its checkpoint SHA-256 is `2e0ed3a09f7b94dedbf55ac2f8d8b2d8c648d3d472bbe909c468c661462d89cc`.
  Local copy: `artifacts/colab_import/rnn_v4_dailydialog.pt` and matching `.json`.
  This public-only checkpoint is not the task-trained provider checkpoint.
- Weights and full run files are ignored by Git. Share the task checkpoint
  privately with E; do not assume a fresh clone includes it.

## Load and export

```powershell
# Optional: choose another verified task checkpoint before starting Streamlit.
$env:REPLYCOACH_V4_CHECKPOINT = "artifacts/role_c_runs/20261020_colab_handoff/v4_task.pt"
.\.venv\Scripts\python.exe -m streamlit run src/replycoach/demo_app.py

# On E's machine, after copying weights: use a NEW output filename.
.\.venv\Scripts\python.exe scripts/export_dev_outputs.py --generator v4 --mode all --output experiments/dev_v4_handoff_01.jsonl
```

`generate(message, context)` takes the original incoming message and selected
chronological memory and returns two model-decoded `Candidate`s, source `V4`,
styles `STANDARD` and `LOCAL`, `item_id=None`. Training uses the same M2 memory
selection. Input IDs are BOS, explicit style token, latest flattened
context/message words, EOS; truncation retains the style. Limits are 64 source
tokens and 40 target tokens. Vocabulary only expands from the 190 TRAIN items;
validation text never adds tokens or supplies optimizer updates.

Decoding is bounded greedy inference in evaluation mode. PAD, UNK, BOS and style
tokens are suppressed; EOS is suppressed only on the first output step. No saved
reply bank or smoke weights are used by the provider. Missing/invalid checkpoints
raise a Role C error instead of generating fake replies. Restart Streamlit after
changing checkpoints because its resource cache retains a loaded generator.

## Score and limitations

The input score is `exp(mean raw log p(token | generated prefix, input))`,
including EOS if emitted. Scores use the original softmax distribution before
decoder-token suppression. They are finite in [0, 1], vary by generation and are
NOT calibrated probabilities of correctness, naturalness or style. Metadata
records checkpoint hash, epoch, decoding settings, token limits, generated token
count, EOS termination, mean log probability and dataset fingerprints.

Do not compare these scores numerically with BM25/cosine as accuracy. The shared
reranker remains unchanged. In this actual validation export hybrid selected
retrieval for all 100 final replies; generator-only used 28 fallbacks. B/E need to
investigate scaling and quality on validation before making a hybrid-win claim.

## Results and verification

See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for measured losses, overlap
diagnostics, limitations and 20 actual validation cases. V4's best task loss is
4.161940 at epoch 8 versus V3's 4.202263 at epoch 10. V4 has lower loss, but its
STANDARD reference-overlap F1 is lower and outputs still repeat or answer the
wrong intent. Different wording for 34/50 style pairs does not establish style
quality. Keep this as an evaluated baseline, not a claim of a finished coach.

The historical DailyDialog files share 99 normalized complete conversations.
Original weights were reused as requested; their historical public validation
numbers must be reported with this contamination limitation. New cleaned copies
are prepared for a deliberate future retrain, not silently substituted for the
data used by the imported checkpoints. The frozen team split is unchanged.

Executed the complete IMPORT + task fine-tuning + generation + export pipeline,
not a new end-to-end DailyDialog training run. A separate verifier reloaded both
public and task weights, recomputed losses, checked fingerprints, and reproduced
all 100 saved outputs per model. The locked 60 test items were not read.

The real Streamlit app was also exercised with the deployed checkpoint and the
reviewed train bank: Hybrid produced 6 retrieval and 2 generation candidates,
and selecting a reply changed only the display composer. Evidence is in
`ui_verification.json`. A five-conversation/three-mode anonymous packet is ready
under `experiments/judging/rolec_handoff_20261020/`; it has 30 candidates and no
ratings or model calls. E should give the judge only `judge_inputs.jsonl`, never
`PRIVATE_mapping.jsonl`.

Final offline test run after deploying the real weights: **254 passed** in
138.53 seconds. Log: `pytest_verified.txt` in the run folder. Missing-model tests
explicitly inject an absent checkpoint path and remain independent of local
weights; the actual model's behavior is checked separately by the verifier and
Streamlit exercise above.

```powershell
.\.venv\Scripts\python.exe scripts/run_role_c.py --run-dir artifacts/role_c_runs/reproduction_01 --reuse-pretrained
.\.venv\Scripts\python.exe scripts/verify_role_c.py artifacts/role_c_runs/reproduction_01
.\.venv\Scripts\python.exe scripts/summarize_role_c.py artifacts/role_c_runs/reproduction_01 --output artifacts/role_c_runs/reproduction_01/report.md
```

Use the recorded environment, identical input/checkpoint bytes and CPU thread
settings for numerical comparisons. This workspace's `.venv` uses existing local
packages through `--system-site-packages`; it is not an independently provisioned
clean environment. Versions and source/data/weight hashes are recorded. Seeds
alone cannot guarantee identical results across hardware/library versions.

## V5 and E evaluation

Role D's `load_v5()` remains unavailable. No V4/V5 winner or invented V5 scores are
reported. Once D supplies a real task-trained model, E should export V5 on the
same 50 queries/bank/M2 configuration, prepare matched anonymous packets and run
the existing local persona judging workflow. C does not replace E's rubric or
use AI judgments as training labels. No held-out test tuning is permitted.
