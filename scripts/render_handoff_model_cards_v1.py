"""Render checkpoint-specific Hugging Face cards in the team's previous section order."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
ROLE=ROOT/'roles/D_transformer'


def section(text,name):
    match=re.search(r'^## '+re.escape(name)+r'\n(.*?)(?=^## |\Z)',text,re.M|re.S)
    if not match:raise ValueError(name)
    return match.group(1).strip()


def main():
    for kind,stem in [('mt5','mt5_reference_A'),('sealion','sealion_repair25')]:
        summary=(ROLE/(stem+'_MODEL_CARD.md')).read_text()
        mt5=kind=='mt5'
        base='google/mt5-base' if mt5 else 'aisingapore/SEA-LION-v1-3B'
        name='ReplyCoach mT5 A (`rewrite_s1`, seed 1, final step 784)' if mt5 else 'ReplyCoach SEA-LION English→Singlish (`repair25`)'
        bundle='replycoach_mt5_A_v1' if mt5 else 'replycoach_sealion_repair25_v1'
        model_type='Encoder–decoder mT5-base with one shared LoRA adapter' if mt5 else 'Decoder-only SEA-LION v1 learned-position MPT with two switchable LoRA adapters'
        description=('An English-first reply generator and Singlish rewriter using the historical shared-intent seed-1 adapter. The export contains the fully adapted mT5 base and the final step-784 adapter; the earlier epoch-4 minimum-loss checkpoint is a different model snapshot.' if mt5 else
            'An English-first reply generator and Singlish rewriter using one original SEA-LION base. The English adapter is the selected epoch-2 full-data task checkpoint from job356177. The Singlish adapter is the earlier anchored renderer followed by selected repair step25 from job357651. Only one adapter is active per stage.')
        licence=('Upstream mT5-base is Apache 2.0; its notice is included. Fine-tuned checkpoint release terms remain unresolved because the training-data and team-publication terms have not all been settled.' if mt5 else
            'The included SEA-LION upstream base/code notice is MIT. The project adapters\' release terms remain unresolved because source-data and team-publication terms have not all been settled.')
        lineage=('Google mT5-base → full span adaptation → full speaker-labelled public reply adaptation → shared task LoRA. The included adapted base is required.' if mt5 else
            'Pinned SEA-LION v1 3B revision836890428766e94322a4b7de27fb081a604703a9 → public LoRA pass → project task adaptation. An independent adapter copy then received English-anchored rewriting and contrastive repair. The original pretrained backbone remained frozen in these project stages.')
        data=('''| Stage | Data | Size |
| --- | --- | --- |
| 0 | CoSEM + NUS SMS span-corruption text | 96,295 chunks:89,638 +6,657 |
| 1 | CoSEM LOCAL + DailyDialog STANDARD reply pairs | 156,052 pairs:80,000 +76,052 |
| 2 | Reviewed derivative of original190 conversations | 190 English +190 LOCAL_REWRITE +180 unchanged history replies =560 rows |

The task targets include14 assistant English corrections, Gemma4 12B LOCAL drafts and117 assistant revisions of those drafts. They are a separate derivative; the original team-authored190 records remain unchanged. This checkpoint uses generated/assistant-edited supervision. Earlier non-synthetic V5 cards describe different checkpoints.'''
            if mt5 else '''| Stage | Data | Size |
| --- | --- | --- |
| Public LoRA | Filtered CoSEM + DailyDialog contextual pairs | 117,634 pairs:53,691 +63,943;400 public validation pairs |
| Project task LoRA | Original190 team conversations +600 corrected expansion records | 790 conversations;1,580 English/direct-LOCAL targets;100 validation targets |
| Anchored rewrite | English/LOCAL pairs with compatible intentions | 745 training;42 validation;45 training and8 validation pairs excluded from rewriting only |
| Contrastive repair | Ordinary replay plus factual positive/negative pairs | Pools of790 English or745 rewrite rows;128 synthetic training pairs and32 validation pairs |

The600 expansion examples originated as500 Gemma4 12B and100 assistant-authored records, then received assistant editorial review. The128/32 factual pairs were assistant authored and counterbalanced across availability, completion, ownership, return direction, exact time, amounts, rooms and before/after constraints. Validation holds entity/value combinations within the same templates. These are not independent human labels.

The selected renderer is at repair update25: it has seen200 ordinary examples and25 auxiliary pairs during repair, not all128 available pairs. The English repair selected update0 and therefore contributes no new repair training to the exported English adapter. Original190 team records remain unchanged.''')
        preprocessing=('Speaker-labelled histories retain actual user/peer/other roles. STANDARD inputs request English; LOCAL_REWRITE inputs include the same conversation and the chosen English draft. During training/CE validation the anchor is the authored/reviewed English target; inference uses the generated English. Source/target limits are192/48 tokens; the shipped loader rejects over-budget inputs without silent truncation. Stage0 uses15% span corruption with mean span3; stage1 uses source/target128/48.' if mt5 else
            'Public pairs use a generic DIALOGUE tag with conservative context/contact/export-artifact filters; DailyDialog punctuation is detokenized. The117,634 pairs are the eligible available export, not the complete unfiltered CoSEM corpus. Project inputs use real speaker labels. Causal prompt tokens are masked from the reply loss, a real EOS is appended, and over-budget inputs are rejected. Rewriting sees the conversation plus its English anchor. Training/validation uses authored English; actual inference uses the generated draft. Total prompt+continuation budget384 tokens; generation cap64.')
        metrics=('Token cross-entropy for each task and their macro average; three separate paired-review judgments for context, chosen-English meaning and natural Singlish; exact-output replay for packaging. Reference-anchored rewrite loss is an oracle-input diagnostic. Uncertain quality judgments do not count as passes.' if mt5 else
            'Ordinary token CE; synthetic positive-reply example-mean CE; normalized positive-versus-negative reply-NLL ranking; contextual usefulness, English→LOCAL meaning and naturalness as separate qualitative factors; exact-output replay for packaging. The32-pair ranking score is teacher-forced and is not generated-reply accuracy. Absence of empty/capped outputs is not correctness.')
        result_summary=('The main reference has28/50 all-three passes in exploratory assistant review, with context35/50, meaning48/50 and Singlish39/50. This supports keeping it as an example reference among the historical rewrite seeds. It establishes neither a global winner across all experiments nor independent human accuracy. No matched V4 result is available.' if mt5 else
            'Selected repair step25 lowers reference-anchored rewrite CE by about0.73% and improves synthetic ranking from26/32 to30/32. Original English is retained. These are promising component diagnostics, while a complete126-case conversational review remains pending. No independent generated-reply accuracy, overall superiority or matched V4 result is claimed.')
        examination=('All50 historical outputs have exploratory assistant review and exact replay; the reviewer also authored target changes and knew earlier outputs. A shared random10 comparison and explicit failures are recorded. No independent interpretability, privacy-memorization or fairness study was performed.' if mt5 else
            'The displayed random10 and the20 frozen context-change development cases have been inspected. The new renderer fixes future-bid/already-bid drift and an offer-to-check role reversal in two random examples. It still preserves incorrect English budget/schedule decisions. A brief aiyo reaction can be contextually valid while failing to preserve its English anchor. This inspection is not a scored independent benchmark.')
        impact=('About3.64 GPU-hours for the recorded historical span/public/task runs, excluding other trials, Google pretraining, loading/inference and export verification. The base was reused across many experiments, so allocating its entire cost to each adapter would double-count shared work.' if mt5 else
            'Recorded successful lineage jobs:5h46m04s public+task,13m18s anchored rewrite,22m50s repair, about6.37 job GPU-hours. Wall times include validation/generation and exclude the matched control, other trials and original SEA-LION pretraining; they are not a measured energy total.')
        architecture=('12 encoder layers and12 decoder layers; width768;12 attention heads; key/value size64; gated-GELU feed-forward2048; vocabulary250,112. Input embeddings are shared, but the stored output head is independent. The compatibility loader preserves that tensor mapping despite legacy tie_word_embeddings metadata. Base582,401,280 + shared LoRA1,769,472 =584,170,752 inference parameters. The base and adapter retain FP32 storage.' if mt5 else
            'Original SEA-LION v1 MPT decoder:32 layers, width2560,20 attention heads, feed-forward expansion4, vocabulary256,000 and learned position embeddings with a2048-position model maximum. Project inference imposes a384-token budget. It uses non-ALiBi attention and the original tuple KV cache. Base3,178,255,360 parameters;10,485,760 per adapter;3,199,226,880 parameters with both resident. Each adapter is rank8/alpha16/dropout0.05 on Wqkv,out_proj,up_proj,down_proj. Only one adapter participates at a time. Verified GPU loading uses a BF16 base and FP32 adapters.')
        citations=('- Xue et al. (2021), mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer. Upstream model: https://huggingface.co/google/mt5-base\n- Chen and Kan (2013), Creating a live, public short message service corpus: The NUS SMS corpus.\n' if mt5 else
            '- AI Singapore, SEA-LION-v1-3B. Upstream model: https://huggingface.co/aisingapore/SEA-LION-v1-3B ; pinned revision is recorded above and in export_recipe.json.\n')
        citations+='- Li et al. (2017), DailyDialog: A manually labelled multi-turn dialogue dataset.\n- Gonzales, Hiramoto, Leimgruber and Lim (2023), The Corpus of Singapore English Messages (CoSEM).\n- No new project paper or DOI is claimed. Existing dataset references do not settle checkpoint distribution terms.'
        text=f'''---
base_model: {base}
library_name: peft
language:
- en
license: other
pipeline_tag: {'text2text-generation' if mt5 else 'text-generation'}
tags:
- lora
- transformers
- singlish
- reply-suggestion
---

# Model Card for {name}

{description}

This is a ReplyCoachSG CS425 Group9 checkpoint handoff. Status: experimental reference/candidate, with no default deployment promotion. It produces user-selectable English and Singlish drafts locally; no prompted teacher or hosted generation API is used at inference.

## Model Details

### Model Description

- **Developed by:** CS425 Group9 (ReplyCoachSG team), Role D
- **Funded by:** Not applicable (course project)
- **Shared by:** CS425 Group9
- **Model type:** {model_type}
- **Languages:** English and Singapore colloquial English (Singlish); other-language quality is unmeasured
- **License:** {licence}
- **Finetuned from:** {lineage}

### Model Sources

- **Repository:** ReplyCoachSG private course repository; training/evaluation report paths are listed in MODEL_CARD.md.
- **Checkpoint:** See the model-details table below and export_recipe.json.
- **Demo/loader:** Included infer.py and ReplyPipeline, rather than the historical V5 default provider.

{section(summary,'Model details')}

## Uses

### Direct Use

{section(summary,'Intended use')}

### Downstream Use

Friend/team comparison and integration into a candidate-generation pipeline. A downstream reranker may be added separately; the shipped inference route does not use one. The CLI returns reply strings, the exact rewrite anchor and generation statuses, not calibrated confidence or a verified probability of correctness.

### Out-of-Scope Use

- Automatic sending without the user selecting/reviewing a draft.
- Reliable factual lookup or decisions where an incorrect reply has material consequences.
- Claims of validated multilingual, demographic or broad dialect coverage beyond the English/Singlish development setting.

Unknown personal plans/preferences may be proposed as selectable drafts under the current project policy. Explicit contradictions, reversed roles and missed intent remain errors; absence of prior evidence alone is not a failure label.

## Bias, Risks, and Limitations

{section(summary,'Limitations')}

### Recommendations

Review contextual usefulness, chosen-English-to-Singlish meaning and naturalness separately. Do not decide success from particles, lower validation loss or fluency alone. Keep users able to select/edit replies, and report the exact checkpoint and decoding configuration in comparisons.

## How to Get Started with the Model

Extract {bundle}.tar and enter the extracted directory. It includes the required base, adapter(s), tokenizer and compatible code; a bare adapter with a different base will not reproduce these examples.

```bash
python -m pip install -r requirements.txt
python infer.py --verify-only
python infer.py --device cuda --verify-examples --output /tmp/{kind}_card_example_check.json
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

{data}

### Training Procedure

#### Preprocessing

{preprocessing}

#### Training Hyperparameters

{section(summary,'Training')}

#### Speeds, Sizes, Times

{section(summary,'Card summary (`docs/licence_checklist.md`)')}

Times, memory units and scope are kept separate: historical per-stage report seconds and SLURM job wall times are not interchangeable. No latency or throughput benchmark was run. Exact archive byte counts and SHA256 values are distributed alongside the archive rather than embedded in a self-referential card.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The50 project development validation conversations were reused throughout experimentation. {'All50 historical rewrite pairs enter the reported mT5 review/CE.' if mt5 else 'The anchored rewrite CE uses42 compatible pairs; English uses50 conversations. Synthetic validation has32 pairs with held entity/value combinations within training templates.'} The locked project test remains unread. The included10 examples are a seeded random draw from45 validation conversations after excluding a previously displayed5, not a new held-out test. Other displayed-example and diagnostic sets are development data.

#### Factors

Contextual appropriateness, faithfulness to the chosen English reply, and recognizably natural Singlish are separate axes. Valid alternative replies, refusals, plans and clarifications are allowed. Assistant-authored labels or judgments are not independent human gold.

#### Metrics

{metrics}

### Results

{section(summary,'Evaluation')}

#### Summary

{result_summary}

## Model Examination

{examination}

## Environmental Impact

- **Hardware:** {'RTX3090 for the recorded adaptation/task runs and export replay' if mt5 else 'L40S for the recorded public/task/rewrite/repair jobs and export replay'}; university on-premises SLURM cluster, Singapore.
- **Recorded compute scope:** {impact}
- **Energy/carbon:** **TODO / not measured.** No CO2 estimate is inferred from a previous checkpoint's card or an assumed power/grid factor.

## Technical Specifications

### Model Architecture and Objective

{architecture}

{'Objectives: span reconstruction, then supervised next-reply/rewrite token cross-entropy. This adapter has no DPO or unlikelihood term.' if mt5 else 'Objectives: causal reply-only supervised CE, followed by low-weight positive/negative factual contrast repair for the selected Singlish adapter. The guard and selection rule are documented above. This is not a DPO/RLHF run.'}

Decoding: deterministic beam4 with no repeated3-grams; maximum{'48' if mt5 else '64'} new tokens per stage. English is generated first, then that exact draft becomes the Singlish anchor. These are not identical token budgets or tokenizers across model families; cross-family CE is not comparable as accuracy.

### Compute Infrastructure

#### Hardware

See the recorded stage/GPU/peak-memory fields above. Export verification: {'job357669, RTX3090,17s' if mt5 else 'job357670, L40S,21s'};10/10 exact paired outputs and no strict-loading or empty/length-cap flags. The export check used isolated Python and Python audit hooks denying original-source-folder opens and network connections; this is a portability/replay check, not an OS security certification.

#### Software

- Python3.13.1; Torch2.14.1+cu130; Transformers5.19.0; PEFT0.21.2.
- Safetensors0.8.0; tokenizers0.23.2; accelerate1.15.0; SentencePiece0.2.2; einops0.8.1; NumPy2.5.3; packaging26.3.
- Exact requirements are in requirements.txt; the loader and compatibility sources are included. Historical training library versions are not all reconstructed; these are the verified export/runtime versions.
- Bundle-relative paths are used at runtime. Original cluster paths in provenance are not required for loading.

## Data, Licences and Privacy

{section(summary,'Data, licences and privacy')}

## Citation

{citations}

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
'''
        (ROLE/(stem+'_README.md')).write_text(text)
        print(stem,len(text.split()),'words')


if __name__=='__main__':main()
