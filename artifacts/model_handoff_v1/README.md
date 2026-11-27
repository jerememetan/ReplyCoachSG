# Checkpoint handoff

Send the two model archives below together with SHA256SUMS. Each archive is a self-contained inference bundle with its exact stored base, adapter(s), tokenizer, compatible loader, training specifications and10 demonstrated example pairs.

Both extracted inference bundles are now available on the project account `CS425G9` under `/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG_checkpoint_handoff_v1/`: `replycoach_mt5_A_v1/` and `replycoach_sealion_repair25_v1/`. SEA-LION was copied on 8 November 2026; all 87 payload files matched the personal-account source checksums. Its personal-account folder and original archives remain as backups. The temporary project-account transfer archive was removed. See the [project transfer record](replycoach_sealion_repair25_v1.project_transfer.json).

| Archive | Selected checkpoint | Size |
|---|---|---|
| [replycoach_mt5_A_v1.tar](replycoach_mt5_A_v1.tar) | mT5 A, shared_intent rewrite_s1 seed1 final784 | 2.35 GB |
| [replycoach_sealion_repair25_v1.tar](replycoach_sealion_repair25_v1.tar) | SEA full English356177 + repair357651 Singlish step25 | 6.45 GB |

Both bundles passed isolated GPU reload:10/10 exact English/Singlish pairs each. Access to original checkpoint folders and network connections was denied during these tests. The packaged base/adapters are byte-identical to their source files, unmerged and unquantized. This verifies transfer/reproduction, not model correctness. SEA is the latest reviewed-sample candidate; a full quality promotion has not occurred.

## Friend setup

On the school cluster, log in as `CS425G9` and use the existing environment through the supplied batch script:

```bash
cd ~/ReplyCoachSG_checkpoint_handoff_v1
sbatch run_both_demo.sbatch
```

This requests one L40S and runs the two models sequentially on the same ten bundled example inputs, checking their saved example outputs. Results go to `demo_outputs/<JOB_ID>/mt5.json`, `sealion.json` and `comparison.csv`; the batch log is `replycoach-demo-<JOB_ID>.log`. To use your own inputs, run `sbatch run_both_demo.sbatch /absolute/path/to/your_inputs.json`. This omits the fixed-example comparison and generates both English and Singlish replies from each model. The script uses the existing project Python environment and SEA-LION's bundled `einops`; no package installation is needed for this cluster setup. Its Slurm configuration can be checked with `sbatch --test-only run_both_demo.sbatch`, which does not submit a job.

Inside each bundle, `infer.py` is the CLI and `ReplyPipeline` loader; `INFERENCE_GUIDE.md` gives usage, `examples/inputs.json` shows the input schema, `examples/expected_outputs.json` contains the demonstration replies, and `MODEL_CARD.md` / `README.md` describe training and limitations. Keep the whole bundle together: the loader needs `base/`, `adapters/`, `src/` and, for SEA-LION, `deps/`.

Tested environment: Python3.13.1, Torch2.14.1+cu130, Transformers5.19.0, PEFT0.21.2. Install a PyTorch build suitable for the friend's machine; the pinned runtime requirements and compatible model implementations are included. GPU inference was replayed on RTX3090 (mT5) and L40S (SEA). CPU FP32 support is implemented but full exported-model CPU replay/performance has not been verified.

```bash
sha256sum -c SHA256SUMS
tar -xf replycoach_mt5_A_v1.tar
tar -xf replycoach_sealion_repair25_v1.tar
```

Inside either extracted model directory:

```bash
python -m pip install -r requirements.txt
python infer.py --verify-only
python infer.py --device cuda --verify-examples --output /tmp/replycoach_example_check.json
python infer.py --device cuda --inputs your_inputs.json --output your_replies.json
```

Use a fresh output filename for each run. The loader refuses to overwrite existing output files. `--verify-only` checks all shipped checksums. `--verify-examples` checks the10 demonstrated pairs in that bundle.

Inputs are a JSON list of objects with `item_id`, `context` (oldest-first text list), `speakers` (matching user/peer/other list), and `message` (incoming peer text). See `examples/inputs.json` after extracting. The output contains English, Singlish, the exact English rewrite anchor and status for each stage.

mT5 is one adapted base with one shared adapter; SEA is one base with separate English and Singlish adapters. Use the included `infer.py` loader to preserve the input format, tokenizer, architecture and adapter switching. The top-level folder is an inference bundle rather than a merged AutoModel directory. Original cluster paths in provenance files are not required at runtime.

No full training data, locked test set, optimizer states or credentials are included. Full source checkpoints remain unchanged. No message or upload to the friend has been sent.


## Model cards

The handoff now follows the team's earlier two-card format. Each archive has a project summary at `MODEL_CARD.md`, a full Hugging Face-style card at `README.md`, populated adapter-component READMEs and the original quickstart at `INFERENCE_GUIDE.md`. Model/loader files retain their verified bytes; documentation and checksums were revised.

- mT5 A: [project summary](../../roles/D_transformer/mt5_reference_A_MODEL_CARD.md) and [Hugging Face card](../../roles/D_transformer/mt5_reference_A_README.md).
- SEA-LION repair25: [project summary](../../roles/D_transformer/sealion_repair25_MODEL_CARD.md) and [Hugging Face card](../../roles/D_transformer/sealion_repair25_README.md).

Use the current SHA256SUMS for these card-bearing archives. The previous archive hashes describe the initial exports without these completed cards.
