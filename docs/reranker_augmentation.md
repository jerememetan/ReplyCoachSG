# V6 abbreviation augmentation

Approved in chat on 27 October 2026: reuse the repository's short-form mapping
to randomly expand the Kaggle classifier's training data. Keep the original
examples, request up to three extra input variants, and leave validation/test
data and candidate replies unchanged. Randomness is seeded and reproducible.

## Design

The existing `ABBR` table in `scripts/data-split/near_dups.py` remains the source.
Packaging reads its literal dictionary without importing/executing that script,
selects a conservative allowlist, reverses it to full-word → short-form choices,
and ships a JSON snapshot plus its hash with the private Kaggle trainer.
Exclude ambiguous `gg`, `gna`, `thr`, `ur` and `anot` entries for this experiment.

A portable augmentation helper runs once before optimization. It changes only
conversation text and the incoming peer message, with replacement probability
0.35 per eligible whole-word occurrence, up to three distinct additional
contexts per original item. A bounded retry loop skips unchanged/duplicate
contexts; examples with no eligible words receive no artificial duplicates.
Speaker labels, style instructions, numbers, Chinese text and replies are
preserved. The same context variants are used across both styles and every
positive/negative pair belonging to the item. Labels are inherited, including
the uncertainty of the existing weak negatives. Originals remain in training.

The notebook exposes copies, probability and augmentation seed. Training without
an augmentation map preserves baseline behavior. Every augmented row records
its original item ID, variant and source label provenance. Saved expanded pairs,
counts, mapping/payload hashes and settings make the run auditable. The 190
underlying conversations do not become independent new reviewed conversations.

## Verification and comparison

Tests must establish stable results despite input reordering, seed variability,
whole-word boundaries, protected formatting/replies, unchanged validation,
duplicate/no-op skipping, valid configuration and actual optimizer consumption.
The packaged standalone trainer must run without the repository on its path.
Retain the completed unaugmented notebook/checkpoint. Publish any augmented run
as a separate private notebook/dataset and keep its checkpoint separate. Compare
on the same unchanged development replay before recommending a checkpoint.
No test examples, new judge calls or automatic demo-default switch are needed.

## Run and usage

- Separate private notebook: [V6 abbreviation augmentation](https://www.kaggle.com/code/tanjingminjereme/replycoach-v6-minilm-fine-tuning-20261027-aug01).
- Separate private dataset: [minimal augmented-run inputs](https://www.kaggle.com/datasets/tanjingminjereme/replycoach-v6-training-20261027-aug01).
- Both privacy flags were read back using the authenticated official MCP. The
  notebook requests a Tesla T4. The original unaugmented notebook is retained.
- The exact existing bundle remains 1,140 training / 300 validation pairs. Before
  optimization, augmentation expands training to **2,556 pairs**: 852 positive
  and 1,704 weak-negative pairs. There are 236 extra contexts across 121 of the
  190 original items. Unchanged/duplicate contexts are skipped, so this is not
  the theoretical fourfold maximum. No augmented context exactly matches a
  validation query. This check does not establish semantic independence.
- Local standalone smoke: actual pinned MiniLM, two CPU optimizer updates,
  explicitly nonproduction. Expanded-pair count/hash and unchanged validation
  were verified. **384 tests passed in 109.47 seconds.**

The training cell in `notebooks/E_reranker_kaggle.ipynb` exposes:

```python
AUGMENTATION_COPIES = 3
REPLACEMENT_PROBABILITY = 0.35
AUGMENTATION_SEED = 425
```

These are extra variants, not total copies. Replacements are attempted with
probability 0.35; unchanged/duplicate outcomes are discarded and retried within
a bound. A different seed samples different variants. The generated dataset
already contains the portable helper and mapping snapshot: the notebook needs
no repository checkout. Set copies to 0 to retain the exact original pair order,
or omit the augmentation flags from the standalone trainer to reproduce the
unaugmented training path. Each saved run requires a fresh output directory.

The checkpoint manifest records `augmentation`, expanded/original pair counts,
the mapping hash and source-script hash, seed, rate, requested variant count,
and the hash of `training_pairs.jsonl` saved alongside the run history. Original
item and donor IDs remain on every row; inherited-label provenance identifies
augmented rows. These audit outputs are private and ignored by Git.

For local standalone reproduction after packaging:

```powershell
.\.venv\Scripts\python.exe scripts/train_reranker.py --bundle artifacts/reranker_v6/bundle_20261027_01 --output artifacts/reranker_v6/new_augmented_run --augmentation-map artifacts/reranker_v6/kaggle_20261027_aug01/dataset/abbreviations.json --augmentation-copies 3 --augmentation-probability 0.35 --augmentation-seed 425
```

Production training requires CUDA. CPU checks must explicitly add `--smoke`
and optionally `--max-steps 2`; their checkpoints cannot be used as production
V6. The original installed checkpoint and demo default remain unchanged.

## Completed GPU training

The separate Kaggle notebook completed version1 on Tesla T4, four epochs and
640 optimizer updates. Epoch2 was retained (320 updates at that point). It used
the exact same initial model weights and original train/validation pair hashes
as the unaugmented experiment. The downloaded expanded JSONL is byte-identical
to the local preview, with matching manifest hash. Training, validation and
checkpoint saving took 31.22 seconds after model loading, excluding notebook
startup, installation and downloads.

| Epoch | Training pair loss | Validation pair loss | Total updates |
| --- | ---: | ---: | ---: |
| 1 | 0.58002 | 0.62051 | 160 |
| 2 | 0.34584 | **0.55064** | 320 |
| 3 | 0.22352 | 0.72414 | 480 |
| 4 | 0.15232 | 0.88131 | 640 |

The augmented checkpoint's best weak-pair validation loss is worse than the
original V6's 0.48777. More rows also produce more updates per epoch (160 versus
72), so this is an augmentation experiment at a fixed epoch count, not a matched
optimizer-step budget. Neither pair loss nor the larger row count establishes
better reply selection. The saved checkpoint is kept separately under ignored
`artifacts/reranker_v6/kaggle_output_20261027_aug01/reranker_v6/best/`.

## Matched reply-selection replay

After downloading, the local loader verified the actual tensor-weight hash and
replayed the same 400 saved proposals on the same 50 development conversations.
The same 4,000 existing AI-persona ratings were joined after selection; their
SHA-256 and all packet/source-export hashes were checked unchanged. No judge
calls or test-set access were needed.

| Selector | Standard appropriateness /5 | Local appropriateness /5 | Standard would-send | Local would-send |
| --- | ---: | ---: | ---: | ---: |
| Original unaugmented V6 | 3.528 | 3.878 | 55.4% | 68.6% |
| Augmented V6 | 3.442 | 3.926 | 54.6% | 70.6% |
| Fixed V5 generator | 3.666 | 4.110 | 61.2% | 76.8% |

Augmentation changed seven Standard and five Local selections. Paired mean
appropriateness changes versus the unaugmented checkpoint were Standard
**-0.086 [-0.260, 0.052]**, Local **+0.048 [-0.108, 0.240]**. Would-send changes
were -0.8 percentage points [-6.0, 3.4] and +2.0 points [-2.4, 8.0]. Intervals
are descriptive 95% conversation-bootstrap intervals, 20,000 resamples, seed425.
They include zero; this single run does not establish an overall improvement.
Keep the original installed V6 checkpoint active and retain augmentation as a
documented optional experiment. Fixed V5 still has higher point estimates.

These are development results on frozen source outputs and one correlated AI
persona judge, not independent human ratings or held-out test evidence. The
same validation set selected the checkpoints by weak-pair loss. One seed and
unequal optimizer-step budgets further limit interpretation. The experiment
also does not separately measure robustness on a new abbreviation stress set.
It expands spelling coverage, not the set of reviewed scenarios or confirmed
negative labels. It does not establish that augmentation can never help.

Raw results: `artifacts/reranker_v6/replay_augmented_20261027_01/result.json` and
`artifacts/reranker_v6/augmentation_comparison_20261027.json`. The private run
folder includes the paired-analysis script, package receipt, exact abbreviation
snapshot, privacy readbacks, full test output, smoke log and replay log. The
original baseline checkpoint/replay and all source data remain intact.
