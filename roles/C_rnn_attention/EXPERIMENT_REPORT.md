# Role C experiment results

Run: `artifacts/role_c_runs/20261020_colab_handoff`.

## Execution and data boundary

Reused the existing V3 checkpoint and the original Colab/Drive V4 checkpoint, both from epoch 5. DailyDialog pretraining was NOT repeated end to end in this run. Actually ran the pilot V3 smoke, task fine-tuning, generation of both styles on all 50 validation conversations, and all three system modes. Reloaded saved weights to recompute validation loss and reproduce all 100 outputs per model.

Task inputs: 190 reviewed training items (380 reply pairs); 50 validation items (100 reply pairs). Vocabulary expansion uses training text only. M2 selects the same chronological context in training and inference. The locked 60-item test file was not opened or used. No new gold data/labels or hosted model calls were made.

DailyDialog audit found 99 normalized complete conversations shared by historical train/validation files (110 training records, 815 pairs). The script prepares a separate 75,237-pair clean training copy, leaving original datasets and the frozen team split unchanged. Imported weights were trained on the original 76,052 pairs; their historical DailyDialog validation loss is therefore not an independent quality estimate. Task validation is a separate team-authored development set.

## Measured results

| Model | Saved public validation loss (rechecked) | Task best epoch | Task validation loss | STANDARD token F1 | LOCAL token F1 | Different style pairs / 50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 | 4.170419 | 10 | 4.202263 | 0.1336 | 0.0879 | 28 |
| V4 | 4.057285 | 8 | 4.161940 | 0.1168 | 0.0918 | 34 |

Token F1/exact match are reference-overlap diagnostics, not appropriateness, naturalness, style quality or accuracy. Validation loss is token-averaged teacher-forced cross entropy; lower is better for this objective. The minimum task validation loss is from V4. One seed and a small validation set do not establish a statistically reliable quality win.

Both task runs use Adam 0.001, batch size 32, seed 425, up to 60 epochs, patience 10, minimum validation-loss checkpoint selection, 64 source tokens and 40 target tokens. The style token is preserved during truncation. V3/V4 use the existing 64-dimensional embeddings and 128-unit LSTMs; V4 adds the existing Luong attention with source-padding masking. Teacher forcing is batched without changing the recurrence; output and gradient equivalence to stepwise calculation is tested.

## Integration verification

All 50 queries exported in retrieval-only, generator-only and hybrid modes: 150 complete mode records. Retrieval uses BM25 on the disjoint 190-item bank; all modes use M2 and the same cached candidate pools.

| Mode | Selected retrieval replies / 100 | Selected V4 replies / 100 | Fallback replies / 100 |
| --- | ---: | ---: | ---: |
| retrieval_only | 100 | 0 | 0 |
| generator_only | 0 | 72 | 28 |
| hybrid | 100 | 0 | 0 |

Mode completion proves the sources ran; it does not prove contextual correctness or a hybrid quality gain. The existing 0.6 score + 0.4 overlap reranker and fallback threshold were not tuned here. Role B/E must validate score scales before claiming a hybrid improvement.

Manual inspection of saved outputs found failures beyond reference overlap: ITEM-A-017 answers a time-confirmation with 'yes i will be there soon' instead of addressing the time; ITEM-C-038 asks for text-length feedback but gets 'sure i'll meet you there'; ITEM-E-009 concerns a charger but LOCAL mentions sending a seat. These observations are report analysis only, not new gold/training labels. The model remains a weak experimental baseline.

## V4 versus V5

Pending Role D's trained V5 provider. No V5 scores or final V4/V5 winner are claimed. Supply V4 to the agreed integration interface while E runs matched blinded persona evaluation. Do not substitute smoke mT5 weights, retrieval text or fabricated confidence.

## Checkpoint fingerprints

- `v3_task.pt`: `4ce9887b4c7312847af1bdd5d5f84f17c1090c2b6600cb1185488804d3e4a999`.
- `v4_task.pt`: `a8b4f0e1f554b86926959ce0a5df4c7a297bdaa71fe03ecf91e67afcb93d900f`.

## Reproduction

Python: `3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 16:37:03) [MSC v.1929 64 bit (AMD64)]`. PyTorch: `2.13.0+cpu`. CPU threads: 2. See `environment.json`, `requirements_frozen.txt`, `imported_pretraining.json` (when importing), `audit.json`, task histories, `comparison.json`, `completion.json` and `verification.json` in the run folder. Bitwise training reproduction depends on the same checkpoint bytes, dataset bytes, software/build, hardware and thread settings; seeds alone do not guarantee identical results across CPU/GPU or library versions. Historical Colab training environment is not fully recorded, so its original training trajectory has not been reproduced.

```powershell
.\.venv\Scripts\python.exe scripts/run_role_c.py --run-dir artifacts/role_c_runs/reproduction_01 --reuse-pretrained
.\.venv\Scripts\python.exe scripts/verify_role_c.py artifacts/role_c_runs/reproduction_01
```

Use a fresh run directory. Import defaults: `artifacts/rnn_v3_dailydialog.pt` and `artifacts/colab_import/rnn_v4_dailydialog.pt`, each with its matching `.json` report. Omit `--reuse-pretrained` only for a deliberate new full pretraining experiment on cleaned data; that produces a different experiment and is not expected to recreate imported historical metrics.

## Actual validation generations (20 conversations)

These are saved model outputs on existing human-authored validation items, not new gold labels. Cases include the ten lowest and ten highest mean V4 reference-overlap values; selection by overlap is diagnostic, not a human quality rating.

### ITEM-A-010

Context: this course is so tough. / yeah, I could really use some help too.

Incoming: peer: we very kiasu sia, first lesson already like that

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | yes i am going to the first floor | yes i am sorry i can't find it | haha, its okay |
| LOCAL | ya i can't find it | yes i will be there soon | haha, nehmind |

### ITEM-A-017

Context: Hey! Where are you now? / I am reaching Harbourfront MRT, how about you?

Incoming: Eh i double check with you, we are meeting at 7pm right?

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | yes i am going to the first class | yes i will be there soon | Nope, its at 8pm |
| LOCAL | yes i will be there | yes i will be there soon | No la, its at 8pm |

### ITEM-A-024

Context: Hey have you printed the worksheet given out yesterday? / No, I will need to print it in school. Do you know how to use the printer?

Incoming: paiseh, i don't know how to use the printer in school.

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | ok i will be there on time | we can go through the seat | It's okay, I will figure it out |
| LOCAL | ok i will be there on time | we can go through the seat | It's okay, I will find out how to use it one |

### ITEM-A-038

Context: are we still playing badminton tonight? / yeah, I think so

Incoming: i might be ten mins late leh

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | i am sorry i can't find my work | sure let's go back together | That's okay, we can wait for you. |
| LOCAL | i am sorry i can't find my work | no worries not be done | Can la, we wait for you. |

### ITEM-A-054

Context: We are going to the club later right? / Yeah, I will go back to shower first.

Incoming: Aiyo, don't need la, let's go now

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | okay i will be there soon | okay i will be there soon | Fair point, let's go |
| LOCAL | okay i will be there soon | okay i will be there soon | ok la, let's go |

### ITEM-B-002

Context: I haven't eaten breakfast / same sia, I'm hungry also

Incoming: where you want eat?

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | i am going to the first class | i am sorry i can't make it | How about the food court near the library? |
| LOCAL | i am going to the first class | ya | Can try the food court near the library. |

### ITEM-B-029

Context: I can't remember which room we're in / I'm checking the timetable

Incoming: which room is our class in?

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | can i check for you | yes i will be there for you | We're in SR3-3 today. |
| LOCAL | can i check for you | okay i will be there for you | Today is SR3-2 |

### ITEM-B-042

Context: I'm waiting outside already / I'm reaching soon

Incoming: sorry, I took the wrong bus

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | okay i will be there on time | okay i will be there soon | No worries. Just let me know when you get here. |
| LOCAL | okay i will be there on time | okay i will be there soon | gg bro, update me when you reach. |

### ITEM-B-043

Context: a few of us are going out this weekend / sounds fun

Incoming: you want come?

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | yes i am i am going to the first class | yes i am late | Sure, I'd be happy to join you. |
| LOCAL | ya i will be late | yes i am late | Yea sure |

### ITEM-B-056

Context: are you free for badminton tonight? / I should be free

Incoming: the court is already booked for seven

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | can i check for you | sure i will be there soon | Okay, seven works for me. |
| LOCAL | okay i will be there soon | okay i will be there soon | Ok, seven can. |

### ITEM-C-038

Context: ive finished drafting the conclusion / great

Incoming: can help me check if this part sounds too long?

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | sure i'll send you the time | sure i'll meet you there | sure. send it to me and i'll take a look |
| LOCAL | yes i will be there on time | sure i'll meet you at the entrance | can, just send me bah. i help you check |

### ITEM-A-044

Context: have you submitted the project? / not yet, I am checking it first

Incoming: remember ah, dont blur like sotong

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | yes i am going to the first class | yes i am sorry i can't it i will be there for you | Don't worry, I will remember to submit |
| LOCAL | can i check for you | yes i am sorry i can't it i will be able to find it | Can, I will remember one. |

### ITEM-C-045

Context: we said we'd sit together tmr / yep

Incoming: you still coming early to get seats right?

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | yes i am going to the first class | yes i am going to work on the campus | yes, i'll arrive a little early |
| LOCAL | yes i am going to the first class | ya i will be there soon | ya, i come early and get seats first |

### ITEM-C-058

Context: im heading to the court now / okay, i'll leave soon

Incoming: court still occupied leh, need wait a bit

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | okay i will be there on time | okay i will be there tonight | okay, i'll come over slowly then |
| LOCAL | okay i will be there on time | okay i will be late soon | okay lor, i come a bit later then |

### ITEM-E-015

Context: Hi guys, remember to do the quiz, deadline is next week / yep noted

Incoming: hi have you done the quiz alr anot

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | yes i am i am going to the first class | yes i am going to work on the phone | Yes, I finished it already. |
| LOCAL | yes i am i am going to the first class | yes i will be there for you | Ya i fin alr |

### ITEM-A-035

Context: I got to go for my appointment later / Yeah I should get going too

Incoming: don't forget to dabao your food later

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | okay i will be there on time | okay i will be there tonight | Okay, thank you! |
| LOCAL | okay i will be there on time | okay i will be late soon | Okay, gamxia |

### ITEM-D-007

Context: u wanna try the matcha at luckin? / nahh

Incoming: then what u wanna drink later?

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | i am going to the first floor | sure what time would you like to eat | I would like to try the new cafe latte there. |
| LOCAL | i am going to the first class | can i book one for you | i want drink the new latte |

### ITEM-D-033

Context: hi / i am already at the bus stop

Incoming: wya

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | yes i am going to the library | yes i am going to the bus | I am currently still on the way to the bus stop. |
| LOCAL | yes i will be there on time | okay i will be there soon | im still otw |

### ITEM-A-030

Context: How long is our booking? / Oh no, our timer went off.

Incoming: sian, need book somewhere else, our booking ran out

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | can i send them later | okay i will be there soon | okay, i will book a pod at the library |
| LOCAL | can i check the details | ok i will be there soon | okay i will chope a pod at the library |

### ITEM-D-029

Context: eh so how, you still want bid for the morning class? / yah can

Incoming: ok then rmb to bid before bidding ends

| Style | V3 | V4 | Human reference |
| --- | --- | --- | --- |
| STANDARD | ok i will be there on time | ok i will be there for you | Ok, I will bid later. |
| LOCAL | ok can | ok i will be there tonight | ok i bid later |

