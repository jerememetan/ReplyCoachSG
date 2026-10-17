# Role A: fixed split and blind-rating protocol

## 1. What was done

- **Gate:** 300 items, 60 per author, all `completed`, all `privacy_ok`, review chain A<-E, B<-A, C<-B, D<-C, E<-D intact, no duplicate IDs.
- **Near-duplicates:** every pair scored on message + last two turns + STANDARD reply (`scripts/near_dups.py`).
  Pairs scoring >= 0.42 are grouped automatically; 6 more groups were added by hand
  (`data/split/review/manual_near_dup_groups.json`). A group is always kept on one side of every split.
- **Split:** `scripts/make_split.py`, seed 425, run once, seed not tuned. Group-aware simulated annealing to match
  proportions of author, scenario, intent, number of local terms and Mandarin/pinyin mix. Exactly 12 test items per author.
- **Validation:** 50 items inside the 240 dev items (10 per author), chosen the same way. Everything else in dev is the
  reply bank / training pool (190).

| file | items | `split` value | use |
|---|---|---|---|
| `data/dev/train.jsonl` | 190 | `dev` | reply bank, training |
| `data/dev/validation.jsonl` | 50 | `dev` | shared validation queries for all experiments |
| `data/test/test.jsonl` | 60 | `test` | locked; final evaluation only |

Record contents are unchanged except the `split` field. The frozen assignment is `data/split/manifest.json`. Re-running
`make_split.py` with the manifest present only re-applies it, so fixing an item's text never reshuffles the split.

## 2. Freeze checklist (do in this order)

1. Fix the flagged items in `data/authoring/items_*.jsonl` (see `data/split/review/audit_issues.csv`, errors first).
   Fix in place; do not delete items.
2. `python scripts/make_split.py --items-dir data/authoring --out-dir .` (re-applies the manifest; re-run `scripts/verify_split.py`).
3. `python scripts/export_dev_outputs.py --check-only`
4. Commit, tag `data-v1`, and paste the SHA-256 values from `data/split/split_report.md` into the team chat.
5. Announce: nobody edits these files again; nobody inspects, tunes on or trains with `test.jsonl`.
   Consider keeping `test.jsonl` off the shared branch until the final evaluation.

Message for the team chat:
> Official split is frozen (data-v1). Use `data/dev/validation.jsonl` (50 items) as the ONLY validation set for every
> comparison (BM25 vs MiniLM, M0/M1/M2, V4 vs V5, hybrid). Index only `data/dev/train.jsonl` as the reply bank / training
> data. Do not open or use `data/test/test.jsonl`. Hashes are in data/split/split_report.md.

## 3. Blind human rating

**What is rated:** the same 50 validation items, every system's STANDARD and LOCAL reply (BM25, MiniLM, V4, V5, hybrid
as available) plus the team-authored reference (`REF`, hidden). Identical text from two systems is shown once.

**Who:** the five team members. Nobody rates an item they authored or reviewed; each item gets 2 raters (set automatically).

**Workload:** rows per rater ~ (items x 2 styles x distinct replies per style) x 2 / 5. With 5 systems and 50 items that is
about 200 rows each; at ~30 s per row that is ~100 minutes. For less, use `--items 30`.

**Steps**
1. Produce each system's output on `validation.jsonl` with `scripts/export_dev_outputs.py` (new filename each run).
   Check one exported row, then map its field names with `--id-field/--standard-field/--local-field`.
2. Calibration (everyone rates the same 5 items, scores discarded, discuss disagreements, fix wording in `rating_rubric.md`):
   ```
   python scripts/make_rating_kit.py --queries data/dev/validation.jsonl --items 5 --calibration \
     --system "BM25=experiments/retrieval_dev_run1.jsonl:mode=retrieval_only" --raters "<5 names>" --out-dir rating_kit_calib
   ```
3. Main round (sheets are `rating_kit/rater_sheets/<name>.xlsx`; send each rater only their own file):
   ```
   python scripts/make_rating_kit.py --queries data/dev/validation.jsonl \
     --system "BM25=...:mode=retrieval_only" --system "MiniLM=..." --system "V4=..." --system "Hybrid=...:mode=hybrid" \
     --raters "<5 names>" --out-dir rating_kit_dev1
   ```
   Keep `PRIVATE_rating_key.csv` and `PRIVATE_assignments.csv` with Role A only. Do not commit them.
4. Collect the returned `.xlsx` files, then:
   ```
   python scripts/analyze_ratings.py --key rating_kit_dev1/PRIVATE_rating_key.csv --sheets "returned/*.xlsx" --out rating_report
   ```
   Check agreement first (alpha < 0.4: recalibrate before reading system scores).
5. The 60 test items are rated once, after all systems are frozen (`--final-test-ack` is required).

Ratings are human only; the kit never generates scores (no LLM labels).
