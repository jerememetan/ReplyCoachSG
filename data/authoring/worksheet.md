# Authoring worksheet

Each member writes their own items. Do not copy the pilot fixtures.

## Final quota per person

- 60 original items saved as `unassigned`
- Independently peer-review one teammate’s set (60 items)

The review cycle is A reviews B, B reviews C, C reviews D, D reviews E, and E
reviews A. Each item gets one independent peer review. Saving a valid review
automatically sets that item's `status` to `completed`. After all 300 items are
completed, Role A assigns a fixed, balanced 80:20 split: 240 development and 60
held-out test items.

## Scenario menu

Use one of these nine topic-only scenarios:

1. `meals_and_food`
2. `study_sessions_and_spaces`
3. `course_materials`
4. `assignments_and_group_projects`
5. `classes_and_tutorials`
6. `meetups_and_transport`
7. `social_events_and_outings`
8. `sports_and_exercise`
9. `campus_services_and_equipment`

The app also assigns one of nine incoming-message intents. Read the definitions and boundary rules in [`docs/annotation_guide.md`](../../docs/annotation_guide.md) before writing. You may change an automatically suggested combination when it does not produce a natural exchange.

## Split rule

Authors never assign splits. Role A assigns official `dev` / `test` values after review using the fixed split. If an item was exposed during model development, it cannot stay in test.
