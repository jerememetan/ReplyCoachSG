# Model card — ReplyCoach SEA-LION English→Singlish (repair step 25)

Role D owns this card. It follows the earlier project-card format. Numbers come from the named run reports; **TODO** means not measured. This is the experimental pipeline shown in the latest comparison and exported for friend/team inspection.

## Card summary (`docs/licence_checklist.md`)

| Field | Value |
| --- | --- |
| Parameter count | 3,178,255,360 base; 10,485,760 per adapter (0.33% of base); 3,199,226,880 with both adapters resident; one adapter active at a time |
| Training time | Completed jobs: public+project 5h46m04s; original rewrite 13m18s; repair branch 22m50s. These wall times include validation/generation; excluded controls and other experiments are not counted |
| GPU type | NVIDIA L40S, university `crimson` SLURM cluster |
| Peak memory | Original rewrite job recorded 7,212,123,136 allocated bytes (~6.72 GiB) and 10,838,081,536 reserved bytes (~10.09 GiB); full public/task and repair peak GPU allocation **TODO** |
| Beat V4 on the same development set? | **TODO**: no matched V4 comparison for this pipeline |
| Status | Selected experimental Singlish renderer; full 126-conversation quality review pending; no default promotion |

## Model details

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

## Training

All project adaptation stages train LoRA only; the original SEA-LION backbone stays frozen.

| Stage | Data | Settings | Selection |
| --- | --- | --- | --- |
| Public adaptation, job 356177 | 117,634 filtered contextual pairs: 53,691 CoSEM + 63,943 DailyDialog; 400 public validation pairs | 1 full pass; LR 5e-5; microbatch 1 × accumulation 16; generic `DIALOGUE` tag for both sources | Final full-pass adapter, after 7,353 optimizer updates |
| Project task, same job | 790 conversations: 190 original + 600 corrected generated; 1,580 English/LOCAL targets; 100 validation targets | LR 1e-4; microbatch 1 × accumulation 8; maximum 8 epochs, patience 2; stopped after epoch 4 | Epoch 2 by equal-style reference CE; 2.553996 macro |
| Original anchored rewrite, job 356662 | 745 eligible English/LOCAL pairs; 42 validation pairs | Copy of selected English adapter; LR 3e-5; effective batch 8; maximum 8 epochs, patience 2; stopped after epoch 3 | Epoch 1; rewrite CE 2.355030 |
| Contrastive repair, job 357651 | Ordinary replay pools: English 790, rewrite 745; 128 new assistant-authored positive/negative pairs, 32 validation pairs | Both stages ran 200 updates; LR 1e-5; 8 ordinary examples + 1 auxiliary pair per update; auxiliary weight 0.1; AdamW weight decay 0.01 | English step 0; rewrite **step 25**, not the final step 200 |

The exported rewrite checkpoint has 200 ordinary-example exposures and 25 auxiliary-pair exposures from the repair stage. The 128 pairs describe the available training pool; the selected step-25 checkpoint has not consumed all 128. Both adapters retain their earlier public/project training. They are alternatives on one base, not additive adapters stacked during inference.

Public/task experiment seed is 425; the task phase explicitly resets Torch RNG to 10425 and shuffles with 425+epoch. Rewrite/repair seed is 425. Reply-only cross-entropy is calculated in FP32. Repair adds positive CE and a 0.2-margin ranking term against factual contradictions. Step 0 stays eligible; ordinary validation CE must stay within 2% of its initial value; eligible snapshots minimize `0.8*ordinaryCE + 0.2*synthetic-positiveCE`.

## Evaluation

| Measurement | Before repair | Selected pipeline | Meaning |
| --- | --- | --- | --- |
| English reference CE, 50 development items | 1.969433 | 1.969433 | Original English checkpoint retained |
| Reference-English-conditioned rewrite CE, 42 pairs | 2.355030 | 2.337944 | About 0.73% lower loss, not an accuracy percentage |
| Synthetic positive-reply example-mean CE, 32 pairs | 1.221924 | 0.634814 | Narrow held-entity-combination diagnostic |
| Correct reply ranked above contradictory reply | 26/32 | 30/32 | Teacher-forced reply-NLL ranking, not generated-reply accuracy |
| Portable export replay | — | 10/10 exact English/Singlish pairs, job 357670 on L40S | Packaging check only |
| Same-set V4 comparison | — | **TODO** | No result established |

All 126 development cases were generated under greedy and beam4/ngram3 with own-English and fixed-baseline-English rewriting; 76 available correct-draft cases also have diagnostic rewrites. No generation failures were flagged. The new renderer changed 56/126 beam replies and 67/126 greedy replies; **these are change counts, not wins**. A comprehensive paired quality review is pending. Project validation was reused, synthetic validation shares templates with training, and no locked project test or independent human benchmark is reported.

## Intended use

- Local, user-selectable standard English and Singlish draft suggestions, and friend/team research comparisons.
- Use the two adapters with the included base, tokenizer, input template and beam implementation.
- Contextual usefulness, English→Singlish fidelity and natural Singlish are judged separately. A brief `aiyo, aiyo` reaction can be contextually acceptable even when it omits the English draft's request.

## Data, licences and privacy

| Resource | Use | Recorded status |
| --- | --- | --- |
| SEA-LION v1 3B | Original pretrained base and upstream implementation | Included upstream MIT notice; this does not settle every downstream data/adapter term |
| CoSEM / DailyDialog | Public LoRA adaptation | CoSEM terms review unresolved; project checklist records DailyDialog CC BY-NC-SA 4.0 |
| Team-authored 190 conversations | Project adaptation | Source records preserved byte-for-byte; team publication approval not established here |
| Corrected 600 expansion records | Project adaptation | Originally 500 Gemma4 12B + 100 assistant authored; subsequently assistant reviewed/edited, not human gold |
| Contrastive corpus | Repair | 128 assistant-authored training pairs + 32 validation pairs across 8 factual categories; not independent human labels |

The fine-tuned checkpoint's release licence remains unresolved. Older project restrictions and later authorized synthetic-target experiments differ; this card does not claim course approval or public redistribution clearance. No full training corpus, locked test set, optimizer state or credentials are included in the export.

## Limitations

- English errors survive faithful rewriting. With 75 dollars available for a 40-dollar booking, the model still says it cannot afford the booking.
- It can accept 19:15 dinner after context explicitly says it cannot leave before 20:45.
- Role confusion remains in the class-update example. LOCAL can omit decisions, requests or time specificity.
- Latest repair fixes two inspected rewrite failures: future bidding no longer becomes already bid; an offer to check no longer becomes a request for the peer to check. These examples do not establish an overall quality gain.
- Singlish is sometimes plain English; abbreviations or particles alone do not establish naturalness.
- Public messages were pattern filtered; complete de-identification and model memorization were not independently audited.
- CPU full-model replay, latency/throughput, calibrated confidence, comprehensive fairness/privacy evaluation, and energy/emissions measurements: **TODO**.

## Reproduce and evidence

Extract the export and follow `INFERENCE_GUIDE.md`. Example check: `python infer.py --verify-examples --device cuda --output /tmp/sealion_card_replay.json`, using a fresh output path. Do not substitute a generic ALiBi MPT model or enable both adapters together.

Private repository evidence: `artifacts/sealion_full_v1/results.json`, `artifacts/sealion_full_rewrite_v1/results.json`, `artifacts/sealion_contrast_v1/repair_results.json`, `experiments/sealion_contrast_v1_results.json`, `experiments/sealion_contrast_v1_recipe.json`, and `experiments/checkpoint_handoff_v1.json`. Training source/data are not shipped; the bundle reproduces inference.
