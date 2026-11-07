---
base_model: aisingapore/SEA-LION-v1-3B
library_name: peft
language:
- en
license: other
pipeline_tag: text-generation
tags:
- lora
- transformers
- singlish
- reply-suggestion
---

# Model Card for ReplyCoach SEA-LION English→Singlish (`repair25`)

An English-first reply generator and Singlish rewriter using one original SEA-LION base. The English adapter is the selected epoch-2 full-data task checkpoint from job356177. The Singlish adapter is the earlier anchored renderer followed by selected repair step25 from job357651. Only one adapter is active per stage.

This is a ReplyCoachSG CS425 Group9 checkpoint handoff. Status: experimental reference/candidate, with no default deployment promotion. It produces user-selectable English and Singlish drafts locally; no prompted teacher or hosted generation API is used at inference.

## Model Details

### Model Description

- **Developed by:** CS425 Group9 (ReplyCoachSG team), Role D
- **Funded by:** Not applicable (course project)
- **Shared by:** CS425 Group9
- **Model type:** Decoder-only SEA-LION v1 learned-position MPT with two switchable LoRA adapters
- **Languages:** English and Singapore colloquial English (Singlish); other-language quality is unmeasured
- **License:** The included SEA-LION upstream base/code notice is MIT. The project adapters' release terms remain unresolved because source-data and team-publication terms have not all been settled.
- **Finetuned from:** Pinned SEA-LION v1 3B revision836890428766e94322a4b7de27fb081a604703a9 → public LoRA pass → project task adaptation. An independent adapter copy then received English-anchored rewriting and contrastive repair. The original pretrained backbone remained frozen in these project stages.

### Model Sources

- **Repository:** ReplyCoachSG private course repository; training/evaluation report paths are listed in MODEL_CARD.md.
- **Checkpoint:** See the model-details table below and export_recipe.json.
- **Demo/loader:** Included infer.py and ReplyPipeline, rather than the historical V5 default provider.

| Field | Value |
| --- | --- |
| Base | Pinned `aisingapore/SEA-LION-v1-3B`, revision `836890428766e94322a4b7de27fb081a604703a9` |
| Architecture | Decoder-only original MPT implementation; 32 layers, width 2,560, 20 heads, vocabulary 256,000; learned positional embeddings, non-ALiBi |
| English checkpoint | Full-data job 356177, selected task epoch 2; English repair selected step 0, so original English retained |
| Singlish checkpoint | Job 356662 renderer epoch 1 → job 357651 repair **step 25** |
| Export | `replycoach_sealion_repair25_v1.tar`; `base/`, `adapters/english/`, `adapters/singlish/` |
| LoRA | Each adapter rank 8, alpha 16, dropout 0.05 on `Wqkv`, `out_proj`, `up_proj`, `down_proj` |
| Precision | BF16 base for verified GPU inference; PEFT adapter weights retain FP32 storage |
| Generation | English first; switch adapter; rewrite the exact English draft with the same conversation |
| Budgets | 384 total prompt+continuation tokens; maximum 64 generated tokens; beam 4; no repeated 3-grams; length penalty 1 |
| Loader | Included `infer.py` / `ReplyPipeline`; compatible learned-position model and legacy-cache beam decoder |
| Full Hugging Face-style card | `roles/D_transformer/sealion_repair25_README.md`; installed as the export's `README.md` |

## Uses

### Direct Use

- Local, user-selectable standard English and Singlish draft suggestions, and friend/team research comparisons.
- Use the two adapters with the included base, tokenizer, input template and beam implementation.
- Contextual usefulness, English→Singlish fidelity and natural Singlish are judged separately. A brief `aiyo, aiyo` reaction can be contextually acceptable even when it omits the English draft's request.

### Downstream Use

Friend/team comparison and integration into a candidate-generation pipeline. A downstream reranker may be added separately; the shipped inference route does not use one. The CLI returns reply strings, the exact rewrite anchor and generation statuses, not calibrated confidence or a verified probability of correctness.

### Out-of-Scope Use

- Automatic sending without the user selecting/reviewing a draft.
- Reliable factual lookup or decisions where an incorrect reply has material consequences.
- Claims of validated multilingual, demographic or broad dialect coverage beyond the English/Singlish development setting.

Unknown personal plans/preferences may be proposed as selectable drafts under the current project policy. Explicit contradictions, reversed roles and missed intent remain errors; absence of prior evidence alone is not a failure label.

## Bias, Risks, and Limitations

- English errors survive faithful rewriting. With 75 dollars available for a 40-dollar booking, the model still says it cannot afford the booking.
- It can accept 19:15 dinner after context explicitly says it cannot leave before 20:45.
- Role confusion remains in the class-update example. LOCAL can omit decisions, requests or time specificity.
- Latest repair fixes two inspected rewrite failures: future bidding no longer becomes already bid; an offer to check no longer becomes a request for the peer to check. These examples do not establish an overall quality gain.
- Singlish is sometimes plain English; abbreviations or particles alone do not establish naturalness.
- Public messages were pattern filtered; complete de-identification and model memorization were not independently audited.
- CPU full-model replay, latency/throughput, calibrated confidence, comprehensive fairness/privacy evaluation, and energy/emissions measurements: **TODO**.

### Recommendations

Review contextual usefulness, chosen-English-to-Singlish meaning and naturalness separately. Do not decide success from particles, lower validation loss or fluency alone. Keep users able to select/edit replies, and report the exact checkpoint and decoding configuration in comparisons.

## How to Get Started with the Model

Extract replycoach_sealion_repair25_v1.tar and enter the extracted directory. It includes the required base, adapter(s), tokenizer and compatible code; a bare adapter with a different base will not reproduce these examples.

```bash
python -m pip install -r requirements.txt
python infer.py --verify-only
python infer.py --device cuda --verify-examples --output /tmp/sealion_card_example_check.json
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
| Public LoRA | Filtered CoSEM + DailyDialog contextual pairs | 117,634 pairs:53,691 +63,943;400 public validation pairs |
| Project task LoRA | Original190 team conversations +600 corrected expansion records | 790 conversations;1,580 English/direct-LOCAL targets;100 validation targets |
| Anchored rewrite | English/LOCAL pairs with compatible intentions | 745 training;42 validation;45 training and8 validation pairs excluded from rewriting only |
| Contrastive repair | Ordinary replay plus factual positive/negative pairs | Pools of790 English or745 rewrite rows;128 synthetic training pairs and32 validation pairs |

The600 expansion examples originated as500 Gemma4 12B and100 assistant-authored records, then received assistant editorial review. The128/32 factual pairs were assistant authored and counterbalanced across availability, completion, ownership, return direction, exact time, amounts, rooms and before/after constraints. Validation holds entity/value combinations within the same templates. These are not independent human labels.

The selected renderer is at repair update25: it has seen200 ordinary examples and25 auxiliary pairs during repair, not all128 available pairs. The English repair selected update0 and therefore contributes no new repair training to the exported English adapter. Original190 team records remain unchanged.

### Training Procedure

#### Preprocessing

Public pairs use a generic DIALOGUE tag with conservative context/contact/export-artifact filters; DailyDialog punctuation is detokenized. The117,634 pairs are the eligible available export, not the complete unfiltered CoSEM corpus. Project inputs use real speaker labels. Causal prompt tokens are masked from the reply loss, a real EOS is appended, and over-budget inputs are rejected. Rewriting sees the conversation plus its English anchor. Training/validation uses authored English; actual inference uses the generated draft. Total prompt+continuation budget384 tokens; generation cap64.

#### Training Hyperparameters

All project adaptation stages train LoRA only; the original SEA-LION backbone stays frozen.

| Stage | Data | Settings | Selection |
| --- | --- | --- | --- |
| Public adaptation, job 356177 | 117,634 filtered contextual pairs: 53,691 CoSEM + 63,943 DailyDialog; 400 public validation pairs | 1 full pass; LR 5e-5; microbatch 1 × accumulation 16; generic `DIALOGUE` tag for both sources | Final full-pass adapter, after 7,353 optimizer updates |
| Project task, same job | 790 conversations: 190 original + 600 corrected generated; 1,580 English/LOCAL targets; 100 validation targets | LR 1e-4; microbatch 1 × accumulation 8; maximum 8 epochs, patience 2; stopped after epoch 4 | Epoch 2 by equal-style reference CE; 2.553996 macro |
| Original anchored rewrite, job 356662 | 745 eligible English/LOCAL pairs; 42 validation pairs | Copy of selected English adapter; LR 3e-5; effective batch 8; maximum 8 epochs, patience 2; stopped after epoch 3 | Epoch 1; rewrite CE 2.355030 |
| Contrastive repair, job 357651 | Ordinary replay pools: English 790, rewrite 745; 128 new assistant-authored positive/negative pairs, 32 validation pairs | Both stages ran 200 updates; LR 1e-5; 8 ordinary examples + 1 auxiliary pair per update; auxiliary weight 0.1; AdamW weight decay 0.01 | English step 0; rewrite **step 25**, not the final step 200 |

The exported rewrite checkpoint has 200 ordinary-example exposures and 25 auxiliary-pair exposures from the repair stage. The 128 pairs describe the available training pool; the selected step-25 checkpoint has not consumed all 128. Both adapters retain their earlier public/project training. They are alternatives on one base, not additive adapters stacked during inference.

Public/task experiment seed is 425; the task phase explicitly resets Torch RNG to 10425 and shuffles with 425+epoch. Rewrite/repair seed is 425. Reply-only cross-entropy is calculated in FP32. Repair adds positive CE and a 0.2-margin ranking term against factual contradictions. Step 0 stays eligible; ordinary validation CE must stay within 2% of its initial value; eligible snapshots minimize `0.8*ordinaryCE + 0.2*synthetic-positiveCE`.

#### Speeds, Sizes, Times

| Field | Value |
| --- | --- |
| Parameter count | 3,178,255,360 base; 10,485,760 per adapter (0.33% of base); 3,199,226,880 with both adapters resident; one adapter active at a time |
| Training time | Completed jobs: public+project 5h46m04s; original rewrite 13m18s; repair branch 22m50s. These wall times include validation/generation; excluded controls and other experiments are not counted |
| GPU type | NVIDIA L40S, university `crimson` SLURM cluster |
| Peak memory | Original rewrite job recorded 7,212,123,136 allocated bytes (~6.72 GiB) and 10,838,081,536 reserved bytes (~10.09 GiB); full public/task and repair peak GPU allocation **TODO** |
| Beat V4 on the same development set? | **TODO**: no matched V4 comparison for this pipeline |
| Status | Selected experimental Singlish renderer; full 126-conversation quality review pending; no default promotion |

Times, memory units and scope are kept separate: historical per-stage report seconds and SLURM job wall times are not interchangeable. No latency or throughput benchmark was run. Exact archive byte counts and SHA256 values are distributed alongside the archive rather than embedded in a self-referential card.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The50 project development validation conversations were reused throughout experimentation. The anchored rewrite CE uses42 compatible pairs; English uses50 conversations. Synthetic validation has32 pairs with held entity/value combinations within training templates. The locked project test remains unread. The included10 examples are a seeded random draw from45 validation conversations after excluding a previously displayed5, not a new held-out test. Other displayed-example and diagnostic sets are development data.

#### Factors

Contextual appropriateness, faithfulness to the chosen English reply, and recognizably natural Singlish are separate axes. Valid alternative replies, refusals, plans and clarifications are allowed. Assistant-authored labels or judgments are not independent human gold.

#### Metrics

Ordinary token CE; synthetic positive-reply example-mean CE; normalized positive-versus-negative reply-NLL ranking; contextual usefulness, English→LOCAL meaning and naturalness as separate qualitative factors; exact-output replay for packaging. The32-pair ranking score is teacher-forced and is not generated-reply accuracy. Absence of empty/capped outputs is not correctness.

### Results

| Measurement | Before repair | Selected pipeline | Meaning |
| --- | --- | --- | --- |
| English reference CE, 50 development items | 1.969433 | 1.969433 | Original English checkpoint retained |
| Reference-English-conditioned rewrite CE, 42 pairs | 2.355030 | 2.337944 | About 0.73% lower loss, not an accuracy percentage |
| Synthetic positive-reply example-mean CE, 32 pairs | 1.221924 | 0.634814 | Narrow held-entity-combination diagnostic |
| Correct reply ranked above contradictory reply | 26/32 | 30/32 | Teacher-forced reply-NLL ranking, not generated-reply accuracy |
| Portable export replay | — | 10/10 exact English/Singlish pairs, job 357670 on L40S | Packaging check only |
| Same-set V4 comparison | — | **TODO** | No result established |

All 126 development cases were generated under greedy and beam4/ngram3 with own-English and fixed-baseline-English rewriting; 76 available correct-draft cases also have diagnostic rewrites. No generation failures were flagged. The new renderer changed 56/126 beam replies and 67/126 greedy replies; **these are change counts, not wins**. A comprehensive paired quality review is pending. Project validation was reused, synthetic validation shares templates with training, and no locked project test or independent human benchmark is reported.

#### Summary

Selected repair step25 lowers reference-anchored rewrite CE by about0.73% and improves synthetic ranking from26/32 to30/32. Original English is retained. These are promising component diagnostics, while a complete126-case conversational review remains pending. No independent generated-reply accuracy, overall superiority or matched V4 result is claimed.

## Model Examination

The displayed random10 and the20 frozen context-change development cases have been inspected. The new renderer fixes future-bid/already-bid drift and an offer-to-check role reversal in two random examples. It still preserves incorrect English budget/schedule decisions. A brief aiyo reaction can be contextually valid while failing to preserve its English anchor. This inspection is not a scored independent benchmark.

## Environmental Impact

- **Hardware:** L40S for the recorded public/task/rewrite/repair jobs and export replay; university on-premises SLURM cluster, Singapore.
- **Recorded compute scope:** Recorded successful lineage jobs:5h46m04s public+task,13m18s anchored rewrite,22m50s repair, about6.37 job GPU-hours. Wall times include validation/generation and exclude the matched control, other trials and original SEA-LION pretraining; they are not a measured energy total.
- **Energy/carbon:** **TODO / not measured.** No CO2 estimate is inferred from a previous checkpoint's card or an assumed power/grid factor.

## Technical Specifications

### Model Architecture and Objective

Original SEA-LION v1 MPT decoder:32 layers, width2560,20 attention heads, feed-forward expansion4, vocabulary256,000 and learned position embeddings with a2048-position model maximum. Project inference imposes a384-token budget. It uses non-ALiBi attention and the original tuple KV cache. Base3,178,255,360 parameters;10,485,760 per adapter;3,199,226,880 parameters with both resident. Each adapter is rank8/alpha16/dropout0.05 on Wqkv,out_proj,up_proj,down_proj. Only one adapter participates at a time. Verified GPU loading uses a BF16 base and FP32 adapters.

Objectives: causal reply-only supervised CE, followed by low-weight positive/negative factual contrast repair for the selected Singlish adapter. The guard and selection rule are documented above. This is not a DPO/RLHF run.

Decoding: deterministic beam4 with no repeated3-grams; maximum64 new tokens per stage. English is generated first, then that exact draft becomes the Singlish anchor. These are not identical token budgets or tokenizers across model families; cross-family CE is not comparable as accuracy.

### Compute Infrastructure

#### Hardware

See the recorded stage/GPU/peak-memory fields above. Export verification: job357670, L40S,21s;10/10 exact paired outputs and no strict-loading or empty/length-cap flags. The export check used isolated Python and Python audit hooks denying original-source-folder opens and network connections; this is a portability/replay check, not an OS security certification.

#### Software

- Python3.13.1; Torch2.14.1+cu130; Transformers5.19.0; PEFT0.21.2.
- Safetensors0.8.0; tokenizers0.23.2; accelerate1.15.0; SentencePiece0.2.2; einops0.8.1; NumPy2.5.3; packaging26.3.
- Exact requirements are in requirements.txt; the loader and compatibility sources are included. Historical training library versions are not all reconstructed; these are the verified export/runtime versions.
- Bundle-relative paths are used at runtime. Original cluster paths in provenance are not required for loading.

## Data, Licences and Privacy

| Resource | Use | Recorded status |
| --- | --- | --- |
| SEA-LION v1 3B | Original pretrained base and upstream implementation | Included upstream MIT notice; this does not settle every downstream data/adapter term |
| CoSEM / DailyDialog | Public LoRA adaptation | CoSEM terms review unresolved; project checklist records DailyDialog CC BY-NC-SA 4.0 |
| Team-authored 190 conversations | Project adaptation | Source records preserved byte-for-byte; team publication approval not established here |
| Corrected 600 expansion records | Project adaptation | Originally 500 Gemma4 12B + 100 assistant authored; subsequently assistant reviewed/edited, not human gold |
| Contrastive corpus | Repair | 128 assistant-authored training pairs + 32 validation pairs across 8 factual categories; not independent human labels |

The fine-tuned checkpoint's release licence remains unresolved. Older project restrictions and later authorized synthetic-target experiments differ; this card does not claim course approval or public redistribution clearance. No full training corpus, locked test set, optimizer state or credentials are included in the export.

## Citation

- AI Singapore, SEA-LION-v1-3B. Upstream model: https://huggingface.co/aisingapore/SEA-LION-v1-3B ; pinned revision is recorded above and in export_recipe.json.
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
