
"""Select 10 additional validation queries for manual relevance scoring."""

import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "data" / "dev" / "validation.jsonl"
SCORING = ROOT / "experiments" / "manual_relevance_scoring.csv"

with VALIDATION.open(encoding="utf-8") as f:
    queries = [json.loads(line) for line in f if line.strip()]

with SCORING.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# A query counts as scored when both reply ratings are filled in
scored_ids = {
    row["item_id"].strip()
    for row in rows
    if row["standard_relevance_0_2"].strip() in {"0", "1", "2"}
    and row["local_relevance_0_2"].strip() in {"0", "1", "2"}
}

# Preserve validation-file order and remove previously scored queries
remaining = []
seen = set()

for query in queries:
    item_id = str(query.get("item_id", query.get("id", ""))).strip()
    if item_id and item_id not in scored_ids and item_id not in seen:
        remaining.append(item_id)
        seen.add(item_id)

if len(remaining) < 10:
    raise SystemExit(
        f"Only {len(remaining)} unscored queries remain."
    )

# Spread the sample across the remaining validation-file order
indices = [
    round(i * (len(remaining) - 1) / 9)
    for i in range(10)
]
selected = [remaining[i] for i in indices]

print("Previously scored queries:", len(scored_ids))
print("Additional queries to score:")
for item_id in selected:
    print("-", item_id)

print("\nFor each selected query, rate all six configurations.")
print("Do not use any held-out test queries.")
