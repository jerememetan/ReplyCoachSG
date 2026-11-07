---
base_model: google/mt5-base
library_name: peft
language:
- en
license: other
pipeline_tag: text2text-generation
tags:
- lora
- transformers
- singlish
- reply-suggestion
---

# Model Card for ReplyCoach mT5 A (`rewrite_s1`, seed 1, final step 784)

An English-first reply generator and Singlish rewriter using the historical shared-intent seed-1 adapter. The export contains the fully adapted mT5 base and the final step-784 adapter; the earlier epoch-4 minimum-loss checkpoint is a different model snapshot.

This is a ReplyCoachSG CS425 Group9 checkpoint handoff. Status: experimental reference/candidate, with no default deployment promotion. It produces user-selectable English and Singlish drafts locally; no prompted teacher or hosted generation API is used at inference.

## Model Details

### Model Description

- **Developed by:** CS425 Group9 (ReplyCoachSG team), Role D
- **Funded by:** Not applicable (course project)
- **Shared by:** CS425 Group9
- **Model type:** Encoder–decoder mT5-base with one shared LoRA adapter
- **Languages:** English and Singapore colloquial English (Singlish); other-language quality is unmeasured
- **License:** Upstream mT5-base is Apache 2.0; its notice is included. Fine-tuned checkpoint release terms remain unresolved because the training-data and team-publication terms have not all been settled.
- **Finetuned from:** Google mT5-base → full span adaptation → full speaker-labelled public reply adaptation → shared task LoRA. The included adapted base is required.

### Model Sources

- **Repository:** ReplyCoachSG private course repository; training/evaluation report paths are listed in MODEL_CARD.md.
- **Checkpoint:** See the model-details table below and export_recipe.json.
- **Demo/loader:** Included infer.py and ReplyPipeline, rather than the historical V5 default provider.

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

## Uses

### Direct Use

- Local, user-selectable English and Singlish reply suggestions for the ReplyCoach course project.
- Friend/team inspection, reproducible comparison, and further experiments using the exact included base and shared adapter.
- A person reviews or edits the draft before sending; plausible new plans/preferences are allowed under the project's selectable-draft policy.

### Downstream Use

Friend/team comparison and integration into a candidate-generation pipeline. A downstream reranker may be added separately; the shipped inference route does not use one. The CLI returns reply strings, the exact rewrite anchor and generation statuses, not calibrated confidence or a verified probability of correctness.

### Out-of-Scope Use

- Automatic sending without the user selecting/reviewing a draft.
- Reliable factual lookup or decisions where an incorrect reply has material consequences.
- Claims of validated multilingual, demographic or broad dialect coverage beyond the English/Singlish development setting.

Unknown personal plans/preferences may be proposed as selectable drafts under the current project policy. Explicit contradictions, reversed roles and missed intent remain errors; absence of prior evidence alone is not a failure label.

## Bias, Risks, and Limitations

- Role reversals, missed questions and generic replies remain. For the charger request, this checkpoint generated `Can, I borrow my charger.`
- Meaning can change during rewriting: a completed-slides English reply became `Can, I finish my part before you finish your part.`
- The same validation set was used repeatedly; no independent test accuracy is available.
- Natural wording, likelihood and beam search do not guarantee contextual usefulness.
- Singlish coverage reflects particular training communities and dated SMS text; no broad dialect/fairness certification was performed.
- Public source corpora contain real messages. Pattern filtering is not a complete privacy guarantee, and memorization was not audited.
- Exact external facts are not verified. Unknown draft details may be selectable; explicit context contradictions remain errors.
- CPU full-model replay, latency, throughput, calibrated confidence, energy and emissions: **TODO**.

### Recommendations

Review contextual usefulness, chosen-English-to-Singlish meaning and naturalness separately. Do not decide success from particles, lower validation loss or fluency alone. Keep users able to select/edit replies, and report the exact checkpoint and decoding configuration in comparisons.

## How to Get Started with the Model

Extract replycoach_mt5_A_v1.tar and enter the extracted directory. It includes the required base, adapter(s), tokenizer and compatible code; a bare adapter with a different base will not reproduce these examples.

```bash
python -m pip install -r requirements.txt
python infer.py --verify-only
python infer.py --device cuda --verify-examples --output /tmp/mt5_card_example_check.json
python infer.py --device cuda --inputs your_inputs.json --output your_replies.json
```

Use a fresh output filename. Inputs are a JSON list of item_id/context/speakers/message objects; context is oldest first and speakers contains matching user/peer/other labels. The incoming message is from the peer. Examples are included.

```python
import json
from infer import ReplyPipeline

pipeline = ReplyPipeline(device="cuda")
rows = json.load(open("examples/inputs.json"))
replies = pipeline.generate_pairs(rows)
```

See INFERENCE_GUIDE.md for setup, runtime requirements and CPU limitations. GPU replay was checked with the pinned runtime; CPU full-model replay and performance have not been verified. No full training dataset is shipped.

## Training Details

### Training Data

| Stage | Data | Size |
| --- | --- | --- |
| 0 | CoSEM + NUS SMS span-corruption text | 96,295 chunks:89,638 +6,657 |
| 1 | CoSEM LOCAL + DailyDialog STANDARD reply pairs | 156,052 pairs:80,000 +76,052 |
| 2 | Reviewed derivative of original190 conversations | 190 English +190 LOCAL_REWRITE +180 unchanged history replies =560 rows |

The task targets include14 assistant English corrections, Gemma4 12B LOCAL drafts and117 assistant revisions of those drafts. They are a separate derivative; the original team-authored190 records remain unchanged. This checkpoint uses generated/assistant-edited supervision. Earlier non-synthetic V5 cards describe different checkpoints.

### Training Procedure

#### Preprocessing

Speaker-labelled histories retain actual user/peer/other roles. STANDARD inputs request English; LOCAL_REWRITE inputs include the same conversation and the chosen English draft. During training/CE validation the anchor is the authored/reviewed English target; inference uses the generated English. Source/target limits are192/48 tokens; the shipped loader rejects over-budget inputs without silent truncation. Stage0 uses15% span corruption with mean span3; stage1 uses source/target128/48.

#### Training Hyperparameters

| Stage | Data | Settings | Selected result |
| --- | --- | --- | --- |
| 0: span corruption, full model | 96,295 chunks: 89,638 CoSEM + 6,657 NUS SMS | Seed 425; 2 epochs; LR 3e-4; batch 32; 500 warmup steps; 15% noise; mean span 3 | Step 6,020; validation CE 2.4593 |
| 1: public replies, full model | 156,052 pairs: 80,000 CoSEM `LOCAL` + 76,052 DailyDialog `STANDARD` | Seed 425; 5 epochs run; LR 5e-4; batch 32; 1,000 warmup steps; source/target 128/48 | Epoch 4, step 19,508; validation CE 3.3347 |
| 2: shared task LoRA | 190 English + 190 English-anchored LOCAL rewrites + 180 original history replies = 560 rows | Seed 1; LR 3e-4; batch 16; 23 epochs requested; fixed cap 784 updates; source/target 192/48 | **Final update 784**, during partial epoch 23; selected for the fixed-budget paired comparison |

Task targets use the separate reviewed derivative `data/distillation/shared_intent_v1_reviewed.json`: 14 assistant English corrections, Gemma4 12B LOCAL drafts, and 117 assistant revisions of those drafts. The original 190 team training records were not edited. This checkpoint therefore **does use generated/assistant-edited training targets**. It has no runtime teacher, DPO or unlikelihood objective.

#### Speeds, Sizes, Times

| Field | Value |
| --- | --- |
| Parameter count | 584,170,752 at inference: 582,401,280 base + 1,769,472 LoRA; only the LoRA parameters trained during the task stage (0.30%) |
| Training time | Historical span stage 36m53s; public reply stage 2h59m33s; this task run 1m46s. About 3.64 GPU-hours for those reported runs, excluding other experiments and export checks |
| GPU type | NVIDIA GeForce RTX 3090, university `crimson` SLURM cluster |
| Peak memory | Span 14.640 GiB; public reply 18.122 GiB; this task run 5.956 GiB, as reported by peak allocated CUDA memory |
| Beat V4 on the same development set? | **TODO**: no matched V4 comparison for this checkpoint |
| Status | Main mT5 example reference; no deployment/default promotion or independent accuracy certification |

Times, memory units and scope are kept separate: historical per-stage report seconds and SLURM job wall times are not interchangeable. No latency or throughput benchmark was run. Exact archive byte counts and SHA256 values are distributed alongside the archive rather than embedded in a self-referential card.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The50 project development validation conversations were reused throughout experimentation. All50 historical rewrite pairs enter the reported mT5 review/CE. The locked project test remains unread. The included10 examples are a seeded random draw from45 validation conversations after excluding a previously displayed5, not a new held-out test. Other displayed-example and diagnostic sets are development data.

#### Factors

Contextual appropriateness, faithfulness to the chosen English reply, and recognizably natural Singlish are separate axes. Valid alternative replies, refusals, plans and clarifications are allowed. Assistant-authored labels or judgments are not independent human gold.

#### Metrics

Token cross-entropy for each task and their macro average; three separate paired-review judgments for context, chosen-English meaning and natural Singlish; exact-output replay for packaging. Reference-anchored rewrite loss is an oracle-input diagnostic. Uncertain quality judgments do not count as passes.

### Results

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

#### Summary

The main reference has28/50 all-three passes in exploratory assistant review, with context35/50, meaning48/50 and Singlish39/50. This supports keeping it as an example reference among the historical rewrite seeds. It establishes neither a global winner across all experiments nor independent human accuracy. No matched V4 result is available.

## Model Examination

All50 historical outputs have exploratory assistant review and exact replay; the reviewer also authored target changes and knew earlier outputs. A shared random10 comparison and explicit failures are recorded. No independent interpretability, privacy-memorization or fairness study was performed.

## Environmental Impact

- **Hardware:** RTX3090 for the recorded adaptation/task runs and export replay; university on-premises SLURM cluster, Singapore.
- **Recorded compute scope:** About3.64 GPU-hours for the recorded historical span/public/task runs, excluding other trials, Google pretraining, loading/inference and export verification. The base was reused across many experiments, so allocating its entire cost to each adapter would double-count shared work.
- **Energy/carbon:** **TODO / not measured.** No CO2 estimate is inferred from a previous checkpoint's card or an assumed power/grid factor.

## Technical Specifications

### Model Architecture and Objective

12 encoder layers and12 decoder layers; width768;12 attention heads; key/value size64; gated-GELU feed-forward2048; vocabulary250,112. Input embeddings are shared, but the stored output head is independent. The compatibility loader preserves that tensor mapping despite legacy tie_word_embeddings metadata. Base582,401,280 + shared LoRA1,769,472 =584,170,752 inference parameters. The base and adapter retain FP32 storage.

Objectives: span reconstruction, then supervised next-reply/rewrite token cross-entropy. This adapter has no DPO or unlikelihood term.

Decoding: deterministic beam4 with no repeated3-grams; maximum48 new tokens per stage. English is generated first, then that exact draft becomes the Singlish anchor. These are not identical token budgets or tokenizers across model families; cross-family CE is not comparable as accuracy.

### Compute Infrastructure

#### Hardware

See the recorded stage/GPU/peak-memory fields above. Export verification: job357669, RTX3090,17s;10/10 exact paired outputs and no strict-loading or empty/length-cap flags. The export check used isolated Python and Python audit hooks denying original-source-folder opens and network connections; this is a portability/replay check, not an OS security certification.

#### Software

- Python3.13.1; Torch2.14.1+cu130; Transformers5.19.0; PEFT0.21.2.
- Safetensors0.8.0; tokenizers0.23.2; accelerate1.15.0; SentencePiece0.2.2; einops0.8.1; NumPy2.5.3; packaging26.3.
- Exact requirements are in requirements.txt; the loader and compatibility sources are included. Historical training library versions are not all reconstructed; these are the verified export/runtime versions.
- Bundle-relative paths are used at runtime. Original cluster paths in provenance are not required for loading.

## Data, Licences and Privacy

| Resource | Use | Recorded status |
| --- | --- | --- |
| Google mT5-base | Original pretrained backbone | Upstream Apache 2.0 notice included; later project adaptations documented separately |
| CoSEM | Span text and LOCAL public pairs | Project terms review remains unresolved; no claim of public redistribution clearance |
| DailyDialog | STANDARD public pairs | Project checklist records CC BY-NC-SA 4.0; downstream checkpoint terms have not been independently settled |
| NUS SMS | Span text | Earlier project card records citation/contact expectations and unresolved formal licensing |
| Team-authored conversations | 190 source inputs and original history replies | Original data preserved; team publication approval not established here |
| Gemma/assistant targets | Reviewed derivative used in task training | Explicit generated-label provenance; not independent human gold |

This card does not resolve the release licence of the fine-tuned checkpoint. Historical project checklist restrictions and later authorized synthetic-target experiments differ; no course-compliance approval is inferred. Raw public text and the full project training data are not included in the handoff.

## Citation

- Xue et al. (2021), mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer. Upstream model: https://huggingface.co/google/mt5-base
- Chen and Kan (2013), Creating a live, public short message service corpus: The NUS SMS corpus.
- Li et al. (2017), DailyDialog: A manually labelled multi-turn dialogue dataset.
- Gonzales, Hiramoto, Leimgruber and Lim (2023), The Corpus of Singapore English Messages (CoSEM).
- No new project paper or DOI is claimed. Existing dataset references do not settle checkpoint distribution terms.

## Glossary

- **STANDARD / LOCAL:** standard English and Singlish draft registers.
- **LOCAL_REWRITE:** Singlish generation conditioned on the chosen English draft and conversation.
- **Context pairs:** extra training replies extracted from original histories; distinct from including history in every input.
- **CE:** reference token cross-entropy, not a reply-quality or accuracy percentage.
- **Replay:** reproducing saved output text with the same checkpoint/settings; not proof that the text is a good reply.

## More Information

See MODEL_CARD.md for the course summary, training_specs.json for structured provenance, INFERENCE_GUIDE.md for loading, verification.json for the completed export replay, and manifest.json for per-file checksums.

## Model Card Authors

CS425 Group9, Role D; documentation compiled with the Codex assistant from recorded experiments. The card does not imply independent human approval of all training examples or evaluation labels.

## Model Card Contact

CS425 Group9, Role D, through the course team. No new personal contact information is supplied.

### Framework versions

PEFT0.21.2; Transformers5.19.0; Torch2.14.1+cu130 (verified export runtime).
