# Model card — ReplyCoach mT5 A (shared-intent seed 1, final step 784)

Role D owns this card. It follows the earlier `MODEL_CARD.md` format. Numbers come from the named run reports; **TODO** means not measured. This card describes the exported reference checkpoint, not the older `spk1_ul025_s425` default.

## Card summary (`docs/licence_checklist.md`)

| Field | Value |
| --- | --- |
| Parameter count | 584,170,752 at inference: 582,401,280 base + 1,769,472 LoRA; only the LoRA parameters trained during the task stage (0.30%) |
| Training time | Historical span stage 36m53s; public reply stage 2h59m33s; this task run 1m46s. About 3.64 GPU-hours for those reported runs, excluding other experiments and export checks |
| GPU type | NVIDIA GeForce RTX 3090, university `crimson` SLURM cluster |
| Peak memory | Span 14.640 GiB; public reply 18.122 GiB; this task run 5.956 GiB, as reported by peak allocated CUDA memory |
| Beat V4 on the same development set? | **TODO**: no matched V4 comparison for this checkpoint |
| Status | Main mT5 example reference; no deployment/default promotion or independent accuracy certification |

## Model details

| Field | Value |
| --- | --- |
| Checkpoint | Training job 351130, `shared_intent_v1_seeds12/artifacts/rewrite_s1/final`, seed 1, update 784 |
| Export | `replycoach_mt5_A_v1.tar`; includes `base/` and `adapters/shared/` |
| Base | `google/mt5-base` → full Singlish span adaptation → full speaker-labelled public reply adaptation |
| Adapter | Rank 16, alpha 32, dropout 0.1, attention `q` and `v`; one shared adapter for both stages |
| Input/output | Speaker-labelled conversation → English; same conversation plus that exact English draft → Singlish |
| Budgets | 192 source tokens; 48 new tokens per stage; beam 4; no repeated 3-grams; deterministic decoding |
| Loader | Exported `infer.py` / `ReplyPipeline`; preserves the stored independent mT5 output head |
| Confidence | No calibrated confidence score is returned by the exported CLI |
| Full Hugging Face-style card | `roles/D_transformer/mt5_reference_A_README.md`; installed as the export's `README.md` |

## Training

| Stage | Data | Settings | Selected result |
| --- | --- | --- | --- |
| 0: span corruption, full model | 96,295 chunks: 89,638 CoSEM + 6,657 NUS SMS | Seed 425; 2 epochs; LR 3e-4; batch 32; 500 warmup steps; 15% noise; mean span 3 | Step 6,020; validation CE 2.4593 |
| 1: public replies, full model | 156,052 pairs: 80,000 CoSEM `LOCAL` + 76,052 DailyDialog `STANDARD` | Seed 425; 5 epochs run; LR 5e-4; batch 32; 1,000 warmup steps; source/target 128/48 | Epoch 4, step 19,508; validation CE 3.3347 |
| 2: shared task LoRA | 190 English + 190 English-anchored LOCAL rewrites + 180 original history replies = 560 rows | Seed 1; LR 3e-4; batch 16; 23 epochs requested; fixed cap 784 updates; source/target 192/48 | **Final update 784**, during partial epoch 23; selected for the fixed-budget paired comparison |

Task targets use the separate reviewed derivative `data/distillation/shared_intent_v1_reviewed.json`: 14 assistant English corrections, Gemma4 12B LOCAL drafts, and 117 assistant revisions of those drafts. The original 190 team training records were not edited. This checkpoint therefore **does use generated/assistant-edited training targets**. It has no runtime teacher, DPO or unlikelihood objective.

## Evaluation

All project-quality numbers below use the same 50 reused development validation conversations. The locked project test remains unread.

| Measurement | Result | Interpretation |
| --- | --- | --- |
| Final reference CE | English 2.1749; reference-English-conditioned rewrite 3.8045; macro 2.9897 | Loss diagnostic, not conversational accuracy |
| Earlier minimum reference CE | 2.5333 at epoch 4/update 140 | Different checkpoint; not the final adapter shipped here |
| Assistant paired review | Context 35/50; same meaning 48/50; natural Singlish 39/50; all three 28/50 | Exploratory assistant judgments; uncertain cases are not passes |
| Other rewrite seeds, same recipe | All-three passes 23/50 for seed 425 and 23/50 for seed 2 | Supports a practical reference choice, not a global best-model claim |
| Historical replay | 50/50 exact saved pairs for this checkpoint in jobs 357554 and 357652 | Reproducibility only |
| Portable export replay | 10/10 exact English/Singlish pairs, job 357669 on RTX 3090 | Packaging check, not accuracy |
| V4 comparison | **TODO** | No matched result available |

The reviewer authored substantial target edits, knew previous model behaviour, and reused these validation conversations. Hidden model labels did not establish independent human blinding. The 28/50 result is not a final test score.

## Intended use

- Local, user-selectable English and Singlish reply suggestions for the ReplyCoach course project.
- Friend/team inspection, reproducible comparison, and further experiments using the exact included base and shared adapter.
- A person reviews or edits the draft before sending; plausible new plans/preferences are allowed under the project's selectable-draft policy.

## Data, licences and privacy

| Resource | Use | Recorded status |
| --- | --- | --- |
| Google mT5-base | Original pretrained backbone | Upstream Apache 2.0 notice included; later project adaptations documented separately |
| CoSEM | Span text and LOCAL public pairs | Project terms review remains unresolved; no claim of public redistribution clearance |
| DailyDialog | STANDARD public pairs | Project checklist records CC BY-NC-SA 4.0; downstream checkpoint terms have not been independently settled |
| NUS SMS | Span text | Earlier project card records citation/contact expectations and unresolved formal licensing |
| Team-authored conversations | 190 source inputs and original history replies | Original data preserved; team publication approval not established here |
| Gemma/assistant targets | Reviewed derivative used in task training | Explicit generated-label provenance; not independent human gold |

This card does not resolve the release licence of the fine-tuned checkpoint. Historical project checklist restrictions and later authorized synthetic-target experiments differ; no course-compliance approval is inferred. Raw public text and the full project training data are not included in the handoff.

## Limitations

- Role reversals, missed questions and generic replies remain. For the charger request, this checkpoint generated `Can, I borrow my charger.`
- Meaning can change during rewriting: a completed-slides English reply became `Can, I finish my part before you finish your part.`
- The same validation set was used repeatedly; no independent test accuracy is available.
- Natural wording, likelihood and beam search do not guarantee contextual usefulness.
- Singlish coverage reflects particular training communities and dated SMS text; no broad dialect/fairness certification was performed.
- Public source corpora contain real messages. Pattern filtering is not a complete privacy guarantee, and memorization was not audited.
- Exact external facts are not verified. Unknown draft details may be selectable; explicit context contradictions remain errors.
- CPU full-model replay, latency, throughput, calibrated confidence, energy and emissions: **TODO**.

## Reproduce and evidence

For the export, extract the archive and follow `INFERENCE_GUIDE.md`; run `python infer.py --verify-examples --device cuda --output /tmp/mt5_card_replay.json` with a fresh output path.

Private repository evidence: `artifacts/mt5_base_spans.json`, `artifacts/mt5_base_after_spans_spk.json`, `artifacts/shared_intent_v1/seeds12/rewrite_s1.json`, `experiments/shared_intent_v1_replication.json`, `experiments/mt5_reference_pack_v1/replay_results.json`, and `experiments/checkpoint_handoff_v1.json`. Training reproduction requires the private training data and `scripts/v5_shared_intent_seeds12.sbatch`; the inference bundle does not contain that dataset.
