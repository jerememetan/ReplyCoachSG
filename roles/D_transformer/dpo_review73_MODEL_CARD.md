# Experimental DPO adapters — reviewed 73-preference comparison

Status: experimental, **not selected as an improvement**. The existing V5 baseline
and its model card remain unchanged. Full protocol/results are in
[the experiment report](../../experiments/dpo_review73_v1.md).

| Field | Value |
| --- | --- |
| Architecture | mT5-base, rank-16 LoRA on q/v, alpha 32, dropout 0.1 |
| Inference parameters | 584,170,752 |
| Trainable parameters | 1,769,472 (0.30%) |
| Initial model | Existing speaker-labelled stage-1 `mt5_base_after_spans_spk/best` |
| Supervised task data | 190 authored train conversations, 380 style replies + 180 context pairs |
| Additional DPO data | 73 retained training-only comparisons, 37 STANDARD / 36 LOCAL, covering 61 conversations |
| Label provenance | Frozen local Gemma preferences; isolated selected-reply acceptance screen; assistant review of all 94 retained labels, with 3 excluded and 18 borderline held out |
| User review scope | Six individual policy examples, not the 73 training preferences |
| LLM training labels | Yes, for these DPO adapters; original supervised checkpoints remain non-synthetic |
| Paired synthetic conversations | None added |
| Training | Seeds 425/1/2, lr 3e-4, batch 16, cap 8 epochs, patience 2; DPO weight/beta 0.1/0.1 |
| Reference | Fixed stage-1 log-probabilities; new adapter rather than resuming the original stage-2 adapter |
| GPU | RTX 3090, 24 GB; DPO peak allocation ~9.9 GB |
| Validation | Same 50 reused development conversations; no locked test-set evaluation |
| Mean best validation loss | DPO 2.4951, matched supervised control 2.5068 |
| Reply quality | No consistent improvement established in anonymized assistant review |
| Cluster artifacts | `CS425G9:~/ReplyCoachSG/experiments/dpo_review73_v1/artifacts/dpo_s{425,1,2}/best` |

Limitations: very small preference set; assistant-reviewed labels and assistant
validation judgments are not independent human gold; repeated use of the same
validation conversations; inconsistent role, negation and direct-answering
behavior remains. Short-run curves do not rule out memorization. No upgrade to
the default checkpoint or deployment was made. Base-data terms and licences
remain those documented in the original V5 model card; no new source corpus was
introduced by this experiment.
