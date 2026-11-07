# Full DPO pilot evaluation

{
  "preference_pairs": 17,
  "conversations": 8,
  "baseline_human_choice_ranked_higher": 8,
  "dpo_human_choice_ranked_higher": 8,
  "corrected": 1,
  "regressed": 1,
  "relative_margin_improved": 8,
  "mean_validation_dpo_loss": 0.6930220688090605,
  "fresh_assistant_review": {
    "baseline": {
      "pass": 8,
      "fail": 11,
      "uncertain": 1
    },
    "dpo": {
      "pass": 7,
      "fail": 11,
      "uncertain": 2
    }
  },
  "limitation": "17 correlated preference pairs from8 conversations, previously used in SFT selection/human review. No locked test accessed. Sequence probability ranking is not generated reply quality. Fresh20 audit is post-hoc assistant assessment, not blinded human gold. No statistical superiority claim."
}

## All17 held-out human preferences

Positive margin means the model gives the user-chosen reply a higher total sequence log probability. Delta is DPO minus pre-DPO margin.

| # | Conversation | Baseline margin | DPO margin | Delta |
|---|---|---:|---:|---:|
| 1 | ITEM-A-010 | 0.299 | -0.214 | -0.512 |
| 2 | ITEM-A-054 | 1.764 | 2.323 | +0.558 |
| 3 | ITEM-B-002 | 8.384 | 8.923 | +0.539 |
| 4 | ITEM-C-018 | -1.168 | -0.636 | +0.532 |
| 5 | ITEM-D-009 | -1.171 | -1.461 | -0.290 |
| 6 | ITEM-E-028 | -0.809 | -1.145 | -0.337 |
| 7 | ITEM-A-010 | -8.843 | -9.467 | -0.624 |
| 8 | ITEM-B-002 | 4.495 | 4.706 | +0.211 |
| 9 | ITEM-C-018 | -0.068 | 0.375 | +0.443 |
| 10 | ITEM-C-038 | 3.884 | 4.622 | +0.738 |
| 11 | ITEM-D-009 | 0.598 | 0.265 | -0.333 |
| 12 | ITEM-D-033 | 7.273 | 7.657 | +0.384 |
| 13 | ITEM-E-028 | -0.382 | -0.489 | -0.107 |
| 14 | ITEM-A-010 | -4.262 | -5.045 | -0.783 |
| 15 | ITEM-B-002 | -7.785 | -7.935 | -0.151 |
| 16 | ITEM-C-038 | 1.910 | 2.185 | +0.276 |
| 17 | ITEM-D-009 | -4.692 | -5.101 | -0.409 |

## All20 fresh conversations: assistant review

Rubric: relevance to latest message, correct user/peer roles, compatibility with explicit context facts/constraints. Pass allows alternative wording and ordinary selectable personal drafts; uncertain flags borderline responses. No reference text matching.

| Case | Baseline | DPO | Finding |
|---|---|---|---|
| FRESH-DPO-001 | fail | fail | Agrees to5pm despite cannot leave until6:30. |
| FRESH-DPO-002 | fail | fail | Reverses ownership/roles: peer wants to keep the user calculator; user replies they can keep it. |
| FRESH-DPO-003 | fail | fail | Claims report finished although user has not started. |
| FRESH-DPO-004 | uncertain | uncertain | Sympathy is plausible but fails to clearly acknowledge waiting; weak rather than an explicit contradiction. |
| FRESH-DPO-005 | fail | fail | Contradictory yes/closed answer and invents bakery opening status despite admitted uncertainty. |
| FRESH-DPO-006 | fail | fail | Agrees to Sunday despite user unavailable Sunday. |
| FRESH-DPO-007 | fail | fail | Answers where the bottle was lost instead of its colour. |
| FRESH-DPO-008 | pass | fail | Before confirms completed upload, with incidental timing; DPO instead promises future upload despite it already being done. |
| FRESH-DPO-009 | pass | pass | Agrees to return the borrowed umbrella; correct owner and role. |
| FRESH-DPO-010 | fail | fail | Accepts twenty-dollar buffet despite twelve-dollar budget. |
| FRESH-DPO-011 | pass | pass | Correctly states west entrance from context. |
| FRESH-DPO-012 | fail | fail | Offers to teach software despite user not knowing how to use it. |
| FRESH-DPO-013 | pass | pass | Availability after four supports meeting at4:30. |
| FRESH-DPO-014 | fail | fail | Before self-contradicts and changes topic to arrival; after unexplained refusal and new booking promise do not resolve extra guest. |
| FRESH-DPO-015 | fail | uncertain | Before chooses cookies user cannot eat; DPO acknowledges inability but wraps it in an awkward preference. Partial repair, no confident pass. |
| FRESH-DPO-016 | pass | pass | Correctly confirms possession of peer headphones. |
| FRESH-DPO-017 | fail | fail | Agrees to nine despite needing to leave by eight. |
| FRESH-DPO-018 | pass | pass | Offers to check unknown deadline rather than invent one. |
| FRESH-DPO-019 | pass | pass | Acknowledges thanks and confirms poster commitment. |
| FRESH-DPO-020 | pass | pass | Correctly states tickets not bought yet. |

## Interpretation

No demonstrated generalization gain from this pilot. The selected DPO checkpoint improves training preference loss, but held-out sequence ranking does not improve. Fresh contextual inspection reveals persistent role/constraint failures, one clear upload regression and one partial peanut-response repair. This is evidence to collect more varied preference/correction data or improve context conditioning, not to increase epochs blindly. Model remains available for user testing; no automatic acceptance/rejection or deployment change.
