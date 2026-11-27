# SEA-LION B/C: completed 548-case run

Both requested student pipelines completed all four epochs per task and rewrite stage on Blackwell MIG24GB. C finished Nov9 13:04:53SGT; B13:07:46; CPU archive13:08:07. This replaces the original5000-case authoring plan, which the user reduced to the first548accepted cases. No new dictionary data or vocabulary changes were added.

The snapshot retains523new train/11dev/14evaluation conversations, with original split assignments. Alongside768prior audited conversations, each task model trained1291conversations/2582style targets; each rewrite model trained1812pairs. Both use5%linear warmup then cosine, four full epochs, same public post-CoSEM/DailyDialog LoRA initialization and frozen backbone. All epochs retained. Original datasets/model bundles and independent backup remain unchanged; locked test unread.

| Arm | Selected task epoch / CE | Selected rewrite epoch / CE |
|---|---|---|
| B | 2 / 2.15525 | 2 / 1.83103 |
| C | 2 / 2.13397 | 3 / 1.79453 |

Scores average old/new reference CE, not semantic accuracy. C has lower validation loss; there is no established overall generated-quality winner. Four epochs completed even when later validation worsened. Final outputs did not select epochs.

[Full89-case comparison CSV](results/comparison.csv), [JSON](results/comparison.json), [assistant review](results/assistant_review.json), [verification](results/verification.json). Each case contains the source chat plus original/B/C English and actual-anchor Singlish. Fourteen fixed-reference rewrites per new model are separate under results/outputs/{B,C}/rewrite/. The assistant read all89matched cases, all14fixed-reference cases per arm, and11saved direct-generation dev pairs per arm; this is exploratory review, not independent human gold.

B's exact flu/video reply is English=Singlish `ok, will do`, a valid terse match. Both shorten the finished-playlist acknowledgment to `done already`. C preserves headphone ownership where B reverses it and refuses the14-dollar noodles with a9-dollar budget where B accepts. Both still accept a two-hour movie despite an8:10departure after an8pm start, and keep L3 after anL8correction. C changes thisafternoon to tomorrow; B fixed-reference rewrites omit new snacks/checking commitments. Dictionary vocabulary alone would not solve these errors. Identical neutral short replies can be valid; exact-copy counts are descriptive, not a quality score.

Runnable candidate copies are at `/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_bc548_candidates_v1/B` and `/C`, using the unchanged portable infer.py, base and tokenizer, plus each new selected English/Singlish adapter. No merging/quantization or default replacement. Base safetensors are hard-linked to immutable original storage within the same home filesystem; these copies are not independent backups. Export includes fresh provenance and ten deliberately mixed examples, including known failures.

On the personal cluster account:

```bash
module load Python/3.13.1-GCCcore-13.3.0
/common/home/users/h/hm.teoh.2024/ReplyCoachSG_focus_v1/.venv/bin/python -I /common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_bc548_candidates_v1/B/infer.py --device cuda --inputs /absolute/path/inputs.json --output /absolute/path/new_outputs.json
```

Run GPU inference inside a Slurm allocation. Change B to C for the other model. Inputs are a JSON list of item_id, context(string list), speakers(user/peer/other list), message. The loader verifies all bundle hashes and freezes both adapters. It refuses to overwrite an existing output. Inference-only replay job359505 COMPLETEDexit0in52seconds: each candidate reproduced10/10frozen pairs exactly, with empty strict-loading problems and no decoder flags. See [replay verification](results/replay_verification.json). No additional student training.

Complete archive with every saved epoch adapter: `/common/home/users/h/hm.teoh.2024/ReplyCoachSG_sealion_large_bc_v1_results/experiment.tgz`,783265829bytes, SHA230a1b777c634f0aeac683e6f48572272716ad40fc8cc15086995cbf08a63fd7. Optimizer states excluded. Local results_only548_v1.tgz SHA3e13cdfc42c756defbe58327427b649ace53cfa1b52e186bea220a67a8ce1f70 verified230included files,127payload hashes,7frozen data hashes,267actual output pairs and28fixed-reference pairs, including actual English anchors and prompt hashes. No decoder flags. This verifies execution/provenance, not correctness of every reply.
