# User requirements for reply quality

The deployed model remains mT5. Larger models may be offline teachers or research comparators.

The user clarified three primary requirements:

1. **Recognizably Singlish:** the LOCAL draft should sound naturally Singaporean, with enough local phrasing, syntax, vocabulary or pragmatic particles to distinguish it from ordinary standard English. Nearly identical neutral-English wording is not sufficient evidence of successful style control. Naturalness matters; counting or stacking particles is not the objective.
2. **Approximately the same meaning across styles:** interpret the user's phrase “Singlish and local” as STANDARD and LOCAL/Singlish, following the established output labels. This interpretation should be corrected if the user intended another distinction. Preserve the essential actor, recipient, facts, time, quantity, negation, uncertainty, decision, request and commitment. Different wording, local idiom and minor politeness/acknowledgement differences are allowed. Do not require literal translation or identical incidental phrasing.
3. **A coherent conversational response:** both drafts must work as the user's next turn. They may differ from the authored reference and introduce plausible user-selectable details. Accept valid refusals, counterproposals, offers and useful clarifications as well as agreements. Reject explicit contradictions, reversed roles, incoherence and missed intent. Do not invent unstated events simply to excuse an otherwise contradictory reply.

All three requirements matter separately. Strong Singlish cannot compensate for a wrong response or a changed meaning. Fluent, relevant standard English alone does not establish the LOCAL requirement.

## Evaluation consequences

Original-reference token loss is a training/checkpoint diagnostic, not the primary definition of success. A valid alternative can receive higher loss. Model selection must inspect contextual adequacy, recognizable natural Singlish and paired meaning directly.

The existing persona evaluator sees one draft at a time. It can assess individual contextual quality and style, but cannot establish cross-style meaning equivalence. That requires a separate paired review with both drafts visible. Do not claim that the single-draft evaluator already checks all three requirements.

These requirements supersede earlier permissive style anchors that allowed nearly neutral English to receive top LOCAL style marks. Preserve old reports with their frozen prompts/labels; revised style judgments require a fresh, matched evaluation. Existing assistant rankings are exploratory and have not been re-scored under this clarification. No new human example labels or numerical thresholds were supplied with this requirements clarification.
