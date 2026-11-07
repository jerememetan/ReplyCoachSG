# mT5 example references

Use the historical final checkpoint as the main mT5 reference, with its earlier epoch4 checkpoint as an alternative. The recent shared adapter is a comparator, not a demonstrated quality upgrade. These are practical example references, not reliable final-model certification.

The main reference previously passed all three checks on28/50 reused validation conversations in an assistant review, versus23/50 for each of two other shared-intent rewrite seeds. This is not an independent human accuracy score. Its original50 saved English/LOCAL pairs were reproduced exactly in job357554.

The15 examples below are selected for readability and contextual usefulness. They are not random. See `common_validation50.json` for every shared validation conversation, `random_comparison5.json` for a reproducible uniform sample and `known_failures.json` for explicit failures.

## Main mT5 example reference: shared-intent seed1 final step784

Strongest recorded all-three paired review among three replicated shared-intent rewrite seeds:28/50 versus23/50 and23/50. Exact saved pairs reproduced50/50 in job357554.

Checkpoint: `/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG/experiments/shared_intent_v1_seeds12/artifacts/rewrite_s1/final`

Older reviewed derivative of190 project conversations:190 English+190 rewritten LOCAL targets+180 original context pairs; one shared rank16 q/v LoRA. Team-authored source files unchanged.

**ITEM-A-030**

- peer: How long is our booking?
- user: Oh no, our timer went off.
- Incoming: sian, need book somewhere else, our booking ran out
- English: No worries, we'll book somewhere else.
- Singlish: No worries, we book somewhere else.

**ITEM-C-023**

- peer: saturday im quite free
- user: same, no plans yet
- Incoming: want me to jio you for the night market thing?
- English: Sure, what time would you like to go?
- Singlish: Can, what time you want to go?

**ITEM-C-038**

- peer: ive finished drafting the conclusion
- user: great
- Incoming: can help me check if this part sounds too long?
- English: sure, i'll check it for you.
- Singlish: Can, I check for you.

**ITEM-D-038**

- peer: eh wya
- user: i am outside mac
- user: u eh
- Incoming: I am reaching soon, wait for me before going in.
- English: Okay, I'll wait for you.
- Singlish: Okay, I wait for you.

**ITEM-E-025**

- peer: hi class, I had a family emergency and there will not be any class tomorrow.
- user: oh no! what about the in class quiz?
- Incoming: will everyone be free on thursday instead for a make up class instead? paiseh
- English: Okay, I'll be free on Thursday.
- Singlish: Can, Thursday also free.

Known limitations: Borrower/lender roles are reversed; LOCAL says I borrow my charger. Accepts meeting after the peer explicitly declines.

## Alternative mT5 example reference: same seed1 run, epoch4 best-validation

Earlier snapshot of the same training run; complementary coherent examples and lower reference validation loss. Keep alongside final, without asserting better conversational accuracy.

Checkpoint: `/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG/experiments/shared_intent_v1_seeds12/artifacts/rewrite_s1/best`

Same historical190-conversation shared-intent recipe; selected epoch4,140 optimizer updates.

**ITEM-A-030**

- peer: How long is our booking?
- user: Oh no, our timer went off.
- Incoming: sian, need book somewhere else, our booking ran out
- English: Yes, we can book somewhere else.
- Singlish: Can, we book somewhere else.

**ITEM-A-054**

- peer: We are going to the club later right?
- user: Yeah, I will go back to shower first.
- Incoming: Aiyo, don't need la, let's go now
- English: Ok, let's go now.
- Singlish: Can, I go now.

**ITEM-C-045**

- peer: we said we'd sit together tmr
- user: yep
- Incoming: you still coming early to get seats right?
- English: yes, I'm coming early to get seats.
- Singlish: Yes, I coming early to get seats.

**ITEM-C-055**

- peer: lets go to the exhibition at 1
- user: sounds good
- Incoming: can go around 2 instead? i need lunch first
- English: Sure, I'll meet you at 2
- Singlish: Can, I meet you at 2

**ITEM-E-007**

- peer: tomorrow what you want to eat
- user: lets eat at Chinatown tomorrow
- Incoming: onz, tmr meet na li
- English: Okay, I'll meet you at Chinatown tomorrow.
- Singlish: Can, I meet you at Chinatown tomorrow.

Known limitations: Borrower/lender roles are reversed. Invents nine oclock when peer says ten minutes late.

## Recent mT5 comparison reference: seed1 shared adapter without extra context objective

Lowest recent shared-study combined validation CE2.2400; useful current-data comparator. Not established as a conversational upgrade over the historical reference.

Checkpoint: `/common/home/projectgrps/CS425/CS425G9/ReplyCoachSG_mt5_context_control_v1/artifacts/seed1_control/final`

Current corrected790 English+745 compatible LOCAL rewrites, shared rank16 q/v LoRA,784 updates, no extra context loss.

**ITEM-C-001**

- peer: are you going to the library ltr?
- user: yeah, probably after dinner
- Incoming: i'll be at lks around 8, want to meet there?
- English: sure, i'll meet you at lks around 8.
- Singlish: okay, i meet you at lks around 8.

**ITEM-C-038**

- peer: ive finished drafting the conclusion
- user: great
- Incoming: can help me check if this part sounds too long?
- English: I'll check whether the conclusion is too long.
- Singlish: I check if the conclusion too long leh.

**ITEM-C-045**

- peer: we said we'd sit together tmr
- user: yep
- Incoming: you still coming early to get seats right?
- English: yes, I'm coming early to get seats.
- Singlish: yes, I coming early to get seats.

**ITEM-C-055**

- peer: lets go to the exhibition at 1
- user: sounds good
- Incoming: can go around 2 instead? i need lunch first
- English: sure, let's meet at 2
- Singlish: 2 can, meet at 2

**ITEM-D-038**

- peer: eh wya
- user: i am outside mac
- user: u eh
- Incoming: I am reaching soon, wait for me before going in.
- English: ok, i will wait for you.
- Singlish: ok i wait for you

Known limitations: Proposes booking a court that is already booked. Confuses availability for makeup class with checking whether the class is free.

## Reproduce

Use the common fully adapted base and exact tokenizer specified in `manifest.json`. Load the chosen adapter over that base with the checked OriginalMT5 compatibility loader. This is one shared adapter used for both stages. Generate English first, then rewrite that exact draft to LOCAL using the same conversation and speaker labels. Decoding: beam4, no-repeat-ngram3, maximum48 new tokens, maximum192 source tokens, no sampling.

Keep generated-draft and correct-draft diagnostic routes separate. A rewrite can preserve an incorrect English decision. None of these checkpoints changes the project default.

All three checkpoints were replayed in job357652: **50/50 exact saved English/Singlish pairs each**,150/150 total. Original checkpoints were unchanged. This checks reproducibility, not correctness.

Runnable demo (on the cluster where these paths are available; CUDA required):

```bash
PYTHONPATH=src python scripts/demo_mt5_reference_v1.py \
  --reference historical_final \
  --inputs experiments/mt5_reference_pack_v1/common_validation50.json \
  --output /tmp/mt5_reference_demo.json
```

Choose `historical_best` or `current_shared` for the other references. For your own inputs, supply a JSON list of objects with `item_id`, `context` (text list), `speakers` (matching user/peer/other list), and `message`. The script refuses to overwrite an existing output file.

The complete verified architecture, inherited training stages, per-checkpoint datasets, hyperparameters, metrics and sample provenance are in [training_specs.json](training_specs.json). A and B are snapshots of one training run; C is a separate shared-adapter run.
