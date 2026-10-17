
"""Summarise human relevance ratings by retriever and memory mode."""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "experiments" / "manual_relevance_scoring.csv"
OUTPUT_PATH = ROOT / "experiments" / "manual_relevance_summary.csv"

CONFIGS = [
    ("bm25", "M0"),
    ("bm25", "M1"),
    ("bm25", "M2"),
    ("minilm", "M0"),
    ("minilm", "M1"),
    ("minilm", "M2"),
]


def parse_score(value, column, row_number):
    value = value.strip()
    if not value:
        return None

    try:
        score = int(value)
    except ValueError:
        raise SystemExit(
            f"Row {row_number}: invalid {column} value {value!r}"
        )

    if score not in (0, 1, 2):
        raise SystemExit(
            f"Row {row_number}: {column} must be 0, 1, or 2"
        )

    return score


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"Scoring sheet not found: {INPUT_PATH}")

    groups = defaultdict(list)
    seen = set()

    with INPUT_PATH.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "item_id",
            "retriever",
            "memory_mode",
            "standard_relevance_0_2",
            "local_relevance_0_2",
            "coherence_issue",
        }
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise SystemExit(
                f"Missing CSV columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            config = (
                row["retriever"].strip().lower(),
                row["memory_mode"].strip().upper(),
            )
            key = (row["item_id"].strip(), *config)

            if key in seen:
                raise SystemExit(
                    f"Duplicate query/configuration at row {row_number}: {key}"
                )
            seen.add(key)

            if config not in CONFIGS:
                raise SystemExit(
                    f"Unexpected configuration at row {row_number}: {config}"
                )

            standard = parse_score(
                row["standard_relevance_0_2"],
                "standard_relevance_0_2",
                row_number,
            )
            local = parse_score(
                row["local_relevance_0_2"],
                "local_relevance_0_2",
                row_number,
            )

            coherence = row["coherence_issue"].strip().lower()
            if coherence not in {"", "yes", "no"}:
                raise SystemExit(
                    f"Row {row_number}: coherence_issue must be "
                    "'yes', 'no', or blank"
                )

            groups[config].append({
                "item_id": row["item_id"].strip(),
                "standard": standard,
                "local": local,
                "coherence": coherence,
            })

    summary = []

    print("\nHUMAN RELEVANCE SCORING SUMMARY")
    print("-" * 76)

    for retriever, memory_mode in CONFIGS:
        records = groups[(retriever, memory_mode)]

        standard_scores = [
            r["standard"] for r in records
            if r["standard"] is not None
        ]
        local_scores = [
            r["local"] for r in records
            if r["local"] is not None
        ]

        rated_rows = sum(
            r["standard"] is not None and r["local"] is not None
            for r in records
        )
        coherence_rated = [
            r["coherence"] for r in records
            if r["coherence"] in {"yes", "no"}
        ]
        coherence_yes = sum(
            value == "yes" for value in coherence_rated
        )

        def mean(values):
            return sum(values) / len(values) if values else None

        def pct(numerator, denominator):
            return 100 * numerator / denominator if denominator else None

        result = {
            "retriever": retriever,
            "memory_mode": memory_mode,
            "rows": len(records),
            "standard_rated": len(standard_scores),
            "standard_mean": mean(standard_scores),
            "standard_zero_pct": pct(
                sum(s == 0 for s in standard_scores),
                len(standard_scores),
            ),
            "standard_two_pct": pct(
                sum(s == 2 for s in standard_scores),
                len(standard_scores),
            ),
            "local_rated": len(local_scores),
            "local_mean": mean(local_scores),
            "local_zero_pct": pct(
                sum(s == 0 for s in local_scores),
                len(local_scores),
            ),
            "local_two_pct": pct(
                sum(s == 2 for s in local_scores),
                len(local_scores),
            ),
            "coherence_rated": len(coherence_rated),
            "coherence_issue_pct": pct(
                coherence_yes, len(coherence_rated)
            ),
        }
        summary.append(result)

        def show(value):
            return "n/a" if value is None else f"{value:.1f}"

        print(
            f"{retriever.upper():6} {memory_mode}: "
            f"rows={len(records)}, both styles rated={rated_rows}/50, "
            f"STANDARD mean={show(result['standard_mean'])} "
            f"({len(standard_scores)} rated), "
            f"LOCAL mean={show(result['local_mean'])} "
            f"({len(local_scores)} rated), "
            f"coherence issues={show(result['coherence_issue_pct'])}% "
            f"({len(coherence_rated)} rated)"
        )

    with OUTPUT_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nSummary CSV: {OUTPUT_PATH}")
    print("Blank ratings were excluded from averages.")


if __name__ == "__main__":
    main()
