CORRECTED GENERATED EXPANSION V1

Scope: all600 generated additions were inspected against their complete supplied conversation, incoming message, English target and LOCAL target.515 revised,85 retained.195 English and502 LOCAL targets revised;16 examples also needed an explicitly logged synthetic-context/speaker/incoming-message repair. Valid alternative decisions and future proposals are allowed; unknown external facts, role reversals and contradictions are corrected. No global word replacements or automatic particle-count acceptance rule was used.

Team-authored data: data/dev/train.jsonl is unchanged. combined_train.jsonl starts with its exact original bytes (190 records), then the corrected600 additions. Earlier expanded versions had rewritten team targets; those rewrites are NOT used in this combined file. Validation data is unchanged, and the original locked test was not read.

Files:
- generated_corrected.jsonl:600 generated additions with assistant editorial provenance.
- combined_train.jsonl:790 conversations /1580 English+LOCAL reply targets.
- audit.jsonl:one audit entry per generated example, with exact before/after changes and reasons.
- changes.csv:515 changed examples with original/corrected conversations, messages and replies side by side.
- patches.json:explicit per-example editorial patch specification.
- review_record.json:full reviewed-ID coverage, source hashes and limitations.
- manifest.json:counts, integrity hashes and invariants.

These are assistant-reviewed editorial corrections authorized by the user. They are not independently human-rated gold; naturalness and semantic judgements can still contain errors. The85 unchanged examples are recorded as inspected, not silently skipped. Training improvement remains to be evaluated.

Checks:9 regression checks pass for exact team-byte preservation, all600 audit entries, replayable diffs, protection against team edits/invalid role edits, validation separation, hashes and the existing human-review importer boundaries. Actual SEA-LION tokenizer checked all1580 targets under384 tokens, max140, no truncation. No training was restarted with this dataset. Existing jobs and checkpoints retain their frozen inputs. A future training recipe must explicitly point to this new version; do not silently modify an old snapshot or present new labels as human approvals.

Rebuild: python3 scripts/build_generated600_repair_v1.py
Test: .venv/bin/python -m pytest -q tests/test_generated600_repair_v1.py tests/test_human_expansion_review.py
