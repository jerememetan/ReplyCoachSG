# Split report

seed=425  method: group-aware simulated annealing, groups never split, exact sizes 60 test / val_size validation


stratified by: author, scenario, intent, local_terms_count(0/1/2+), code_mix(zh/pinyin)


near-duplicate threshold: 0.42 (+ manual groups)


counts: train 190, validation 50, test 60


## By author
| value | train | validation | test | total |
|---|---|---|---|---|
| Cody | 38 | 10 | 12 | 60 |
| Jereme | 38 | 10 | 12 | 60 |
| Teoh Hao Ming | 38 | 10 | 12 | 60 |
| Tristan Ang Boon Yew | 38 | 10 | 12 | 60 |
| Yangtian | 38 | 10 | 12 | 60 |

## By scenario
| value | train | validation | test | total |
|---|---|---|---|---|
| assignments_and_group_projects | 21 | 6 | 7 | 34 |
| campus_services_and_equipment | 14 | 4 | 5 | 23 |
| classes_and_tutorials | 22 | 6 | 7 | 35 |
| course_materials | 20 | 5 | 7 | 32 |
| meals_and_food | 27 | 7 | 8 | 42 |
| meetups_and_transport | 21 | 5 | 6 | 32 |
| social_events_and_outings | 22 | 6 | 7 | 35 |
| sports_and_exercise | 19 | 5 | 6 | 30 |
| study_sessions_and_spaces | 24 | 6 | 7 | 37 |

## By intent
| value | train | validation | test | total |
|---|---|---|---|---|
| apologise_or_explain | 18 | 5 | 6 | 29 |
| change_or_reschedule | 20 | 5 | 6 | 31 |
| confirm_plan | 20 | 5 | 6 | 31 |
| decline | 18 | 5 | 6 | 29 |
| inform_or_update | 27 | 7 | 8 | 42 |
| invite_or_propose | 21 | 6 | 7 | 34 |
| remind_or_nudge | 14 | 4 | 5 | 23 |
| request_help_or_resource | 23 | 6 | 7 | 36 |
| request_information_or_status | 29 | 7 | 9 | 45 |

## By number of local_terms (0/1/2+)
| value | train | validation | test | total |
|---|---|---|---|---|
| 0 | 78 | 20 | 24 | 122 |
| 1 | 67 | 19 | 22 | 108 |
| 2 | 45 | 11 | 14 | 70 |

## SHA-256
- `data/dev/train.jsonl`  03485c7d6b510711fbdee5f3eac0044e334d9d7a25bf3c5c86665b0646e8eee7
- `data/dev/validation.jsonl`  68ab3fcfd283bddd37b8f4ac93a79edcc20a45b1ab9d12ce9732e26f74b20a52
- `data/test/test.jsonl`  5e9e8f93b3b88a5bfb62949b0b318a2deb4799775647467bf699ad55af426a8e
- `data/split/manifest.json`  3bf9664b1e1e8660df24202eb9b9479d1c5f4b3b2c0bce6155a6b1b335405ef3
