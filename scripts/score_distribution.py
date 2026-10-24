
import json
import statistics
from pathlib import Path

EXPORT_DIR = Path("experiments")

FIELDS = ("input_score", "overlap", "final_score")


def get_scores(candidate):
    """Extract available V6 scores, handling fallback candidates."""
    meta = candidate.get("meta") or {}

    return {
        "input_score": meta.get("v6_input_score"),
        "overlap": meta.get("v6_overlap"),
        "final_score": candidate.get("score"),
    }


def summarise(values):
    """Print descriptive statistics for available numeric scores."""
    values = [float(v) for v in values if v is not None]

    if not values:
        print("  No values available")
        return

    print(
        f"  min={min(values):.4f} "
        f"median={statistics.median(values):.4f} "
        f"mean={statistics.mean(values):.4f} "
        f"max={max(values):.4f} "
        f"(n={len(values)})"
    )


def main():
    files = sorted(EXPORT_DIR.glob("retrieval_*.jsonl"))
    valid_files = []

    for path in files:
        parts = path.stem.split("_")

        if (
            len(parts) == 3
            and parts[0] == "retrieval"
            and parts[1] in ("bm25", "minilm")
            and parts[2].upper() in ("M0", "M1", "M2")
        ):
            valid_files.append((path, parts[1], parts[2].upper()))

    if not valid_files:
        print(f"No matching retrieval exports found in {EXPORT_DIR.resolve()}")
        return

    print(f"Found {len(valid_files)} retrieval export files.")

    for path, retriever, memory in valid_files:
        records = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(
                        f"WARNING: Invalid JSON in {path}, "
                        f"line {line_number}: {exc}"
                    )

        for style in ("standard", "local"):
            candidates = [
                record[style]
                for record in records
                if isinstance(record.get(style), dict)
            ]

            print(
                f"\n{retriever.upper()} {memory} {style.upper()} "
                f"(candidates={len(candidates)})"
            )

            extracted = [
                get_scores(candidate)
                for candidate in candidates
            ]

            for field in FIELDS:
                values = [row[field] for row in extracted]
                available = sum(value is not None for value in values)

                print(f"  {field} (available={available}):", end=" ")
                summarise(values)

    print("\nDone.")


if __name__ == "__main__":
    main()
