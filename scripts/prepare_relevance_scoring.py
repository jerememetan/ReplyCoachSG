
"""Create a manual relevance-scoring CSV from six development exports."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "experiments"
OUTPUT_PATH = EXPORT_DIR / "manual_relevance_scoring.csv"

CONFIGS = [
    ("bm25", "M0"),
    ("bm25", "M1"),
    ("bm25", "M2"),
    ("minilm", "M0"),
    ("minilm", "M1"),
    ("minilm", "M2"),
]

FIELDS = [
    "item_id",
    "retriever",
    "memory_mode",
    "original_message",
    "selected_memory",
    "standard_reply",
    "standard_source",
    "standard_relevance_0_2",
    "local_reply",
    "local_source",
    "local_relevance_0_2",
    "coherence_issue",
    "notes",
]


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Rename or move it first so existing scores are preserved."
        )

    rows_by_config = {}

    for retriever, memory_mode in CONFIGS:
        filename = f"retrieval_{retriever}_{memory_mode.lower()}.jsonl"
        path = EXPORT_DIR / filename

        if not path.exists():
            raise SystemExit(f"Missing export: {path}")

        rows = load_jsonl(path)

        if len(rows) != 50:
            raise SystemExit(
                f"{filename}: expected 50 rows, found {len(rows)}"
            )

        indexed = {}

        for line_number, row in enumerate(rows, start=1):
            required = {"item_id", "message", "standard", "local"}
            missing = required - row.keys()

            if missing:
                raise SystemExit(
                    f"{filename}, row {line_number}: "
                    f"missing fields {sorted(missing)}"
                )

            indexed[row["item_id"]] = row

        rows_by_config[(retriever, memory_mode)] = indexed

    id_sets = [
        set(indexed.keys())
        for indexed in rows_by_config.values()
    ]

    if not all(ids == id_sets[0] for ids in id_sets):
        raise SystemExit(
            "The six exports contain different query IDs."
        )

    output_rows = []

    for item_id in sorted(id_sets[0]):
        for retriever, memory_mode in CONFIGS:
            row = rows_by_config[(retriever, memory_mode)][item_id]
            
            output_rows.append({
                "item_id": item_id,
                "retriever": retriever,
                "memory_mode": memory_mode,
                "original_message": row.get("message", f"MISSING MESSAGE: {row.get('item_id', 'unknown')}"),
                "selected_memory": " | ".join(row.get("memory", [])),
                "standard_reply": row["standard"].get(
                    "candidate_text", ""
                ),
                "standard_source": row["standard"].get("source", ""),
                "standard_relevance_0_2": "",
                "local_reply": row["local"].get(
                    "candidate_text", ""
                ),
                "local_source": row["local"].get("source", ""),
                "local_relevance_0_2": "",
                "coherence_issue": "",
                "notes": "",
            })

    with OUTPUT_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Created: {OUTPUT_PATH}")
    print(f"Queries: {len(id_sets[0])}")
    print(f"Configurations: {len(CONFIGS)}")
    print(f"Scoring rows: {len(output_rows)}")


if __name__ == "__main__":
    main()
