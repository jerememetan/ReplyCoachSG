# Faithful rewrite diagnostic v3

Job **355463** was submitted to the group project partition on one RTX 3090 and verified RUNNING with finite baseline losses. No default or application checkpoint changed.

## What was prepared

- 24 original LOCAL targets from authors A/B/C, with assistant-audited equivalent English anchors.
- 12 original LOCAL targets from authors D/E, excluded from gradient updates here.
- 12 paired/targeted English meaning probes: availability, future versus completed booking, ownership, unfinished versus finished work, reassurance, support, uncertainty and returning a charger.
- Original LOCAL text is unchanged. English-only corrections are recorded in the dataset manifest. No public corpus targets are mixed into this diagnostic.

Review the pairs at `data/training_expansion/rewrite_diagnostic_v3/review.md`. These are assistant judgments, not new human labels. Author separation does not provide topic separation, and previous model exposure to original project data cannot be excluded. The two assertions tested cover original target preservation/split leakage and source isolation from target text; two tests passed. Python compilation and batch syntax checks passed.

## Experiment

Start from the same stage1 mT5 base, using the anchor-only serializer. Full model updates, seed 425, batch 4, AdamW learning rate 0.0001, 300 optimizer steps (50 passes over 24 examples), gradient clipping 1.0. This higher-rate repeated fitting run deliberately asks whether the small set can be learned. It is not a production optimization or a comparison with the previous recipe on equal hyperparameters.

Record train and author-heldout loss every 25 steps, plus actual train/validation/probe generations at steps 0, 100 and 300. Generation uses the same beam4 serializer as the previous run. Store a fixed step300 checkpoint; do not select it using meaning-probe inspection. Normalized exact-match is a memorization indicator only: valid alternate Singlish wording can differ, so it is not a quality grade. No early stopping is used because observing capacity and subsequent divergence is the purpose; validation remains monitored.

## Audit of existing outputs

`experiments/rewrite_diagnostic_v3/matched_development.csv` compares all three new models and four earlier checkpoint/decoding controls on the same 20 development contexts and STANDARD anchors. `audit.json` records assistant issue notes separately from surface statistics.

I flagged 13 STANDARD replies for specific context/meaning problems, including agreeing to a time explicitly unavailable, reversing borrower/lender roles, claiming a proofread is already complete, and irrelevant replies to stress or relief. These are qualitative assistant flags, not human win rates; unflagged examples are not certified good.

On these same 20 cases, the old final beam4 rewriter starts with Can 15 times versus 4/1/1 for new context/anchor/exemplar. But the new models copy the English more often: 11/9/13 normalized copies versus 2. This is evidence of a surface tradeoff, not evidence of a quality win. Several newer rewrites also change negation, time or ownership. The old incumbent fails many of the same meaning controls.

## How the result will guide the next change

1. If train rewrites remain wrong despite fitting effort: inspect tokenization, target alignment, model initialization and decoding before adding more data.
2. If train replies become faithful while author-heldout/probe replies fail: capacity is present on this subset; concentrate on training coverage, compositional contrasts and generalization. This alone cannot prove the failure is entirely data-related.
3. If meaning is preserved but style remains weak: improve the distribution and coverage of native Singlish targets, inspecting more than particle counts.
4. Repair STANDARD context use separately with broader, checked student-conversation demonstrations. A rewriter cannot reliably rescue an incoherent anchor while also being required to preserve its meaning.

Before another production trial, add a truly separate topic/conversation evaluation with human judgments for context appropriateness, meaning preservation and Singlish naturalness. Existing inspected cases remain development material. No RL/DPO run is planned from these weak signals.
