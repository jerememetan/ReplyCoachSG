# Native rewrite v2: completed results

All three Slurm jobs completed with exit code 0. Reports were downloaded locally (ignored from Git because they include public corpus text).

| Mode | Selected epoch | Validation loss | Train loss at selection | Can openings /40 | Normalized copies /40 |
|---|---:|---:|---:|---:|---:|
| anchor_only | 4.5 | 2.3863 | 1.5151 | 3 | 15 |
| exemplar | 6.0 | 2.4219 | 1.3390 | 2 | 21 |
| full_context | 7.0 | 2.4064 | 1.1353 | 8 | 16 |

The loss winner is anchor-only, with a small margin. This is component token likelihood, not a human quality win. All arms show late validation deterioration while training loss falls; the selected best checkpoints precede the final checkpoints. Early stopping fired for anchor-only and exemplar; full-context reached its epoch cap.

Fresh-output inspection found copying, weak style transfer, dropped clauses, tense changes, and polarity reversals. Full-context rewrites “Friday afternoon works for me.” as “Friday afternoon cannot.” All three turn a future room booking into a completed booking. Anchor-only and exemplar turn “No worries” into “No” in several cases. Full-context retains some natural light local phrasing but still adds Can inappropriately. Counts include grammatical Can questions and are not counts of unnatural usage. Normalized copies ignore case/punctuation, and copying can be appropriate on individual short replies.

The frozen STANDARD generator itself also produces incoherent or unhelpful anchors in these new contexts, including “You can leave your stand behind your stand.” Thus English content repair remains necessary alongside style transfer. No defaults have been changed. These 40 assistant-authored conversations have now been inspected and must not subsequently be called untouched final evaluation if used for development. The older 30-case human-review rates are not directly comparable to these new 40 cases.

Next: audit semantic fidelity and context appropriateness separately across matched outputs, compare the incumbent on the same development cases, and build broader faithful content/style training pairs. Additional epochs or RL without a reliable preference signal are not supported by these results.
