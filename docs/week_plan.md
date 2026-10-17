# Remaining plan updated 17 October 2026

The original schedule below is historical. The current priorities and evaluation
method supersede its outdated data targets and manual-rating workload.

The 300-item reviewed split is published: 190 train/reply-bank items, 50 shared
validation queries and 60 untouched test items. Generator providers and real hybrid
quality results remain pending. E's local judging runner exists; main ratings and
independent pilot quality checks remain work, not completed results.

## Current evaluation decision

Following professor approval reported by Jereme, replace human output reviewers
with **ten initial configurable persona-based AI reviewers**. Each keeps the same
persona and rates every compared reply on **appropriateness, naturalness and
style fit**, each 1–5, with an explanation and problem/uncertainty flags. This is
not a three-specialist panel or a human survey. It also records would-send yes/no
and reports its simulated percentage. Jereme/E now owns the full judging workflow;
A retains frozen-data stewardship and supports independent pilot sanity checks.
Dataset authorship and independent
human peer review remain unchanged. See `split_and_rating_protocol.md` for the
current protocol, local command, limitations and remaining evaluation work.

## Remaining milestones

| Dates | Deliverable |
| --- | --- |
| Now–23 Oct | C/D real-model handoffs; B retrieval/memory baselines; E local judge setup and five-conversation/ten-persona validation pilot, with A supporting independent checks; E export handoff and real hybrid verification when a model arrives. Choose the generator using validation quality and compute. |
| 26–30 Oct | Main validation evaluation; analyse persona disagreements and system errors; make evidence-based targeted improvements. Draft the report in parallel. |
| 2–6 Nov | Freeze systems, judge/persona configuration and rubric; evaluate the 60 held-out items once; finish tables, 15–20 qualitative cases, slides and the 20-conversation demo. |
| 7–9 Nov | Rehearsal and blocking bug fixes only. |
| 10 Nov | Final presentation with at least 90% of results complete. |
| By 27 Nov | Final report, code/data/reproduction package and peer evaluations. |

No evaluation results are established by this plan. E must check privacy, the
local judge's reliability and runtime before running the full panel. Do not tune on test data
or use AI ratings as training labels.

## Historical schedule

Snapshot as of 29 Sep 2026. Week 4 Friday was 18 Sep. Status cells and outstanding
data work below describe that earlier point, not current completion.

| Week | Dates (2026) | Friday proof | Status in this repo |
| --- | --- | --- | --- |
| 4 Foundation | 14–18 Sep | V0 retrieves a reply; shared format | Done as a head start |
| 5 Data + first models | 21–25 Sep | Authoring and peer review underway; V1 and V3 outputs | V1 works; V3 one-batch works; team must author data |
| 6 Core upgrades | 28 Sep–2 Oct | 300 items authored; V2, V4, basic V6 | V6 baseline exists; V2/V4 stubs exist; proposal due 2 Oct |
| 7 Generator decision | 5–9 Oct | Choose V4 or V5 | Comparison script still needed |
| 8 Final hybrid | 12–16 Oct | E2E demo on 20 unseen chats | Pipeline already wired; polish later |
| 9 Evaluation | 19–23 Oct | Tables and graphs | Forms to fill |
| 10 Targeted improvement | 26–30 Oct | One retrieval fix, one generator fix | Not started |
| 11 Freeze | 2–6 Nov | Submission-ready | Presentation 10 Nov |
| 12 Buffer | 9–13 Nov | Bugs only | Report 27 Nov |

Historical authoring target (now completed; do not use the 20 fixtures as gold):

- Each person: 60 unassigned items and one independent peer-review set of 60 items
- Review cycle: A reviews B, B reviews C, C reviews D, D reviews E, and E reviews A
- Team total: all 300 items marked `completed` after a valid review
- After all 300 are completed, Role A assigns the fixed, balanced 80:20 split of 240 development and 60 untouched test items
