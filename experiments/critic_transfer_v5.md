# Critic transfer check under user-reviewed policy — 2 November 2026

Job **349274**, output `artifacts/critic_transfer_v5.json`. Source:
`data/training_critic/calibration_transfer_v5.jsonl`. Pairwise rubric:
`selectable-v5-user-reviewed`; isolated rubric: `selectable-single-v3-user-reviewed`.

The six user-reviewed examples clarified that unknown progress, room, deadline
and third-party approval can all be acceptable selectable drafts. Explicit
contradictions and reversed roles are unacceptable. Earlier scores used a
stricter assistant-authored policy and are not current task accuracy.

This run fixes 12 new assistant-authored transfer cases before seeing responses:
four pairs of acceptable unknown details, four pairs targeting explicit
contradictions, two role reversals, one irrelevant definition and one incoherent
reply. Expected flags are inferred from user policy, not individually reviewed by
the user. These are development diagnostics, not independent gold or held-out
test evaluation. Preference expectations remain assistant-authored.

Each case runs pairwise in both orders and judges each draft separately. Keep
candidate-acceptance accuracy separate from ranking and order agreement. Inspect
failures against the actual conversation rather than treating a disagreement as
proof of a model error. Do not tune the prompt during the run.

Only proceed toward a small training-label pilot if the checks support the
selected protocol; never feed these diagnostic cases into DPO. No final test
items are accessed. Do not select a model on old misaligned scores.

## Results

Completed. Isolated verdicts matched all **24/24 candidate labels** and both
candidates in all **12/12 cases**, with no incorrect acceptance, rejection or
uncertainty on this small assistant-authored set. This supports a training-only
pilot, not a claim of real-world accuracy.

Pairwise full checks passed **7/12** cases; order agreement was **10/12**, with
incorrect acceptance flags in **3/12** cases. The remaining issues support
separating acceptance from ranking. Ties, both-bad cases, uncertainty and
order-disagreeing rankings must not become training labels.

A screened 20-comparison training pilot follows with `--screen-single`. Each
kept preference must have an order-consistent, non-uncertain ranking and an
isolated acceptable verdict for its selected draft; uncertainty in either
isolated verdict skips the pair. Isolated judgments are retained in each
question and screening prompt/schema/version in run metadata. Both-bad cases
never automatically promote an unjudged third reference.

25 focused code tests passed, including screening rejection, uncertainty,
order disagreement and saved-label behavior. No DPO training started.
