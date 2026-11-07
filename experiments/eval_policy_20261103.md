# Evaluation-policy alignment diagnostic — 3 November 2026

Job **350486**, completed in **40s** on `toga`. One unchanged first CSV persona (P01), local Gemma4 12B with recorded digest/settings and fresh contexts. The current evaluator rubric is frozen in the raw report.

The project user explicitly reviewed these six drafts. The rubric was changed before the check, so this is a development/policy-alignment diagnostic, not an independent accuracy benchmark, full persona panel or training labels.

Gate fixed before requests: accepted draft → appropriateness at least 4 and would_send true; rejected draft → at most 2 and false; uncertainty fails.

**6/6 policy matches.** The four plausible-detail examples score appropriateness 5/would-send true; the contradiction and role reversal score 1/false. Naturalness/style remain scored independently.

| Example | Expected | Appropriateness | Would send |
| --- | --- | --- | --- |
| USER-REVIEW-001: Halfway through mine. | True | 5 | True |
| USER-REVIEW-002: Room B12. | True | 5 | True |
| USER-REVIEW-003: Sure, six works. | False | 1 | False |
| USER-REVIEW-004: I’ll buy lunch for you. | False | 1 | False |
| USER-REVIEW-005: Friday at 5 pm. | True | 5 | True |
| USER-REVIEW-006: Yes, she approved it this morning. | True | 5 | True |

The model sometimes calls an unverified draft accurate in its explanation. No real-world facts were verified; the scores assess plausible selectable drafts. Old judge runs remain historical and must not be merged with fresh runs using this rubric. The runner records rubric text and rejects changed-rubric resumes.

14 judging-runner tests passed. `scripts/export_dev_outputs.py --check-only` validated 50 queries and a 190-item bank without loading models; that is an input check, not an end-to-end export, model quality result or completed main evaluation.

Raw report: ignored `artifacts/eval_policy_20261103.json`. Updated evaluator rubric: `data/evaluation/judge_rubric.md`; shared anchors: `scripts/data-split/rating_rubric.md`; protocol: `docs/split_and_rating_protocol.md`. Main multi-system/multi-persona validation and final test remain pending.
