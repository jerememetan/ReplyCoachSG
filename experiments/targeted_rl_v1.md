# Targeted RL reward policy preparation

Prepared 2026-11-06; no RL training or automated scorer implemented. User authorized scenario and scoring review before RL. Data data/reward_diagnostics/targeted_rl_v1:18 scenario instances,6 per category (time constraints, availability, ownership). Group-disjoint review/development/heldout partitions each24 drafts over6 scenarios. Each category includes acceptable alternatives/refusals, explicit contradictions/role reversals, and uncertain drafts with reward null. Labels are assistant-authored proposals, not human gold or scorer results. Shared template families limit broad generalization claims. No model/test data read.

Reviewer scripts/rl_reward_review.py on localhost8512. Explicit human decisions save to human_reviews.json, no default selection or blanket approvals. All24 labels remain pending until user review. Development and heldout examples excluded from UI. Next: reconcile human policy corrections; implement and test scoring on fresh model-generated replies with adequate abstention coverage before an RL-vs-baseline pilot. Human policy agreement alone is not reward-scorer validation.

Two tests passed for grouped isolation, truthful provenance, uncertainty mapping, frozen packet preservation and persistent corrected decisions; Streamlit smoke passed.

## Human review completed

All24 decisions saved. Human labels12 acceptable,9 unacceptable,3 uncertain. Three changes from proposals: TIME-0-4 and TIME-1-4 deferral “Let me check and get back to you.” marked acceptable; AVAIL-0-4 “Maybe, I will let you know.” marked unacceptable despite explicit Monday unavailability. AVAIL-1-4 same deferral wording in a different-day scenario remains uncertain, so do not generalize a categorical rule without acknowledging inconsistent scenario judgments. Preserve exact votes, no silent normalization. Proposed development/heldout labels remain assistant-only and predate human policy changes; do not claim these are independently human-approved or adjust heldout labels to force a pass. See targeted_rl_v1_policy_review.json. All three session Streamlit review servers stopped at user request; review files preserved.
