# Raw evaluation ratings

Each archive contains the original JSONL ratings for one run. The separate candidate index lists anonymous candidate and conversation IDs, systems, and reply styles. Candidate IDs are scoped to each run.

| Run | Rating cells | Published summary |
| --- | ---: | --- |
| [Six-system comparison](six_system_20261108/) | 7,200 | [Aggregate JSON](../current_pipeline_20261108.json) |
| [V7 v2 Light Local](v7_v2_local_20261109/) | 600 | [Aggregate JSON](../v7_v2_local_20261109.json) |
| [V7 v3 Light Local](v7_v3_local_20261115/) | 600 | [Results summary](../v7_v3_local_20261115.md) |

Open `ratings.jsonl.gz` with gzip and join `candidate_id` to the corresponding `candidates.csv`. To reproduce the point estimates, average the ten Gemma persona ratings per conversation and then average across conversations for each system and style. For would-send, report the percentage of true values. The ratings come from simulated personas.

The SHA-256 values below apply to the decompressed JSONL bytes.

| Run | SHA-256 |
| --- | --- |
| six_system_20261108 | `401dd757575f2e9d5cf9472a7da061cf90d34d891ba05b4108112e95c25c5865` |
| v7_v2_local_20261109 | `ec20138a66bd969a0e97a589b4346c1640c23fa5912ca2b4ef724e3c7831e72f` |
| v7_v3_local_20261115 | `be708e9f33f2557f59db23e6e1e2e4fb109663f1d2af89cae241b382e3699013` |
