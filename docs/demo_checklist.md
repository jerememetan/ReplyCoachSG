# Role E: 20-conversation demo proof

This is a coverage checklist, **not a completed experiment or authored gold data**.
Use 20 manually selected/written development/demo conversations not used by the
retrieval bank or generator training. Do not inspect the locked 60 test items for
debugging. All final examples and human ratings come from the team.

## Before marking the hybrid proof complete

- A has supplied reviewed, disjoint development train/validation files.
- C/D has supplied the selected task-trained generator with both output styles.
- The demo shows requested mode **Hybrid** and effective mode **Hybrid**, with
  retrieved and generated candidates present. Missing-model fallback is useful
  infrastructure proof, not proof of a working generator/hybrid.
- The team has checked score scaling and documented the model/decoding settings.

## Cases to cover

Record actual incoming text, prior turns, memory/method, selected model, candidate
origins, STANDARD/LOCAL replies, appropriateness/style observations and fallback
reason. Download the demo JSON for traceability; store run outputs outside `data/`.

| Case | Coverage to choose manually | Observation/result |
| --- | --- | --- |
| 01 | Meal invitation | |
| 02 | Meal plan confirmation | |
| 03 | Study-space availability | |
| 04 | Study-plan change | |
| 05 | Course-material request | |
| 06 | Reply to a resource/status update | |
| 07 | Group-project obligation reminder | |
| 08 | Project progress request | |
| 09 | Class/tutorial location | |
| 10 | Class attendance explanation | |
| 11 | Meetup arrival update | |
| 12 | Transport delay/change | |
| 13 | Social invitation | |
| 14 | Polite decline | |
| 15 | Sports/exercise arrangement | |
| 16 | Equipment/help request | |
| 17 | Incoming message with restrained local terms | |
| 18 | Older context matters (compare M0/M1/M2) | |
| 19 | Consecutive messages from the same speaker | |
| 20 | Unknown/low-confidence input and honest fallback | |

## Workflow checks

- Default BM25/MiniLM comparison still gives four clickable suggestions.
- Retrieval-only, generator-only and hybrid use the same selected memory.
- Each reply button fills only the display-only composer; nothing is sent.
- Editing message, conversation, memory, bank, mode or model clears old outputs.
- Missing/failed generation cannot masquerade as a complete hybrid. Retry recovers
  when the real provider becomes available; the original retrieval result survives.
- Long English/Chinese text wraps; keyboard selection/focus remain visible.
- Save examples/screenshots and note limitations. Do not label ranking scores as
  accuracy or turn this developer check into final held-out evaluation.
