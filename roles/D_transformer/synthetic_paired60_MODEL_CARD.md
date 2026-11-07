# Experimental paired-synthetic adapters — 59-conversation pilot

Status: **not selected as an upgrade**. Existing V5 baseline/model card remains
unchanged. Full recipe/results:
[the pilot report](../../experiments/synthetic_paired60_v1.md).

| Field | Value |
| --- | --- |
| Model | mT5-base + rank-16 LoRA on q/v; alpha 32, dropout 0.1 |
| Parameters | 584,170,752 total; 1,769,472 trained |
| Initial checkpoint | Same speaker-labelled stage-1 base as the matched control |
| Original task data | 190 authored conversations, 380 style rows + 180 authored context rows |
| New data | 59 fictional conversations, 118 paired STANDARD/LOCAL reply rows |
| Provenance | Codex assistant author; exact assistant model ID unavailable; local Gemma4 12B pair reviewer and Codex author audit |
| Review scope | Different-model/author review, not independent human/native-speaker validation |
| Human policy input | User reviewed six policy examples; not these 59 conversations |
| Candidate selection | Model approved 60/60; author held out one awkward LOCAL draft without rewriting |
| DPO / unlikelihood | Neither used |
| Training | Seeds 425/1/2; lr 3e-4, batch 16, cap 280 updates, cap 8 epochs, patience 2 |
| Validation | Same 50 development conversations; test set remains locked |
| Mean validation loss | Augmented 2.5308 vs control 2.5068 |
| Primary qualitative review | 16 augmentation vs 19 control preferences, 25 ties, 10 both-bad, 30 identical |
| Primary review limits | Assistant authored data and had seen control outputs; anonymized but not fully blind or independent |
| GPU | RTX 3090 24 GB; allocated peak about 5.5 GB for augmented runs |
| Artifacts | `CS425G9:~/ReplyCoachSG/experiments/synthetic_paired60_v1/artifacts/synthetic_s{425,1,2}/best` |

The added dataset uses mostly standard-English incoming messages; the real chats
have more shorthand, Singlish and code-switching. This is a possible mismatch,
not a proven failure mechanism. Equal optimizer updates do not equalize token
compute, authored-data exposure or checkpoint opportunities. The larger planned
190/570 data-volume arms were not run. This pilot does not establish a general
failure of synthetic supervision. Some roles/direct answers improved while
other answers regressed; no default checkpoint/provider/deployment was changed.

All new situations are fictional. Unknown details may appear as selectable
drafts under the user’s policy; facts_supported is a legacy compatibility flag,
not verification of real-world facts. Source candidates and reviewer reports
remain preserved. Original public-data/base licence records remain those of the
existing V5 model card; no new external corpus was introduced.
