# Role A — Data and evaluation

## Current priority updated 17 October 2026

The 190 train / 50 validation / 60 test split is published. Preserve it. Jereme/E
has taken ownership of the persona judging workflow, initially ten profiles using
local Ollama, the shared three-score rubric and a would-send vote. Do not build a
duplicate runner. Existing Excel scripts remain optional human sanity checks.
This replaces human output raters, not the dataset's independent human review.

By 23 October, coordinate with E on frozen data, privacy and an independent sanity
check of five validation pilot conversations. Support results interpretation and
the later guarded final-test export. The initial local judge must be checked before
main evaluation; personas remain simulated perspectives, not human participants.
Full protocol: `docs/split_and_rating_protocol.md`; current dates: `docs/week_plan.md`.

The Week 4 and later-week outline below is retained as historical context.

## This week (Week 4)

- [x] Annotation guide: `docs/annotation_guide.md`
- [x] Schema: `src/replycoach/schema.py` and `data/schema/`
- [x] Licence checklist: `docs/licence_checklist.md`
- [x] 20-item pilot file: `data/pilot/pilot_20.jsonl`
- [ ] Team review of all 20 items. Rewrite anything you would not send yourself.
- [ ] Fill `TEAM.md`

## How to author

Copy `data/pilot/authoring_template.jsonl`, replace placeholders, keep `privacy_ok` true only if it is true.

## Later

Week 5: coordinate annotation, duplicates, privacy, style, splits.  
Week 6: freeze split, publish stats.  
Week 7: lock test set.  
Week 8: original blind human form (superseded by the persona protocol above).\
Week 9: run eval.  
Week 10: data card.  
Week 11: appendix.

## Friday artefact

A validated JSONL file plus a short note of how many items passed review.
