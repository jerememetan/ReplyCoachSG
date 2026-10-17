
"""Compare retrieval-only exports across retriever and memory modes."""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "experiments"

CONFIGS = [
    ("bm25", "M0"), ("bm25", "M1"), ("bm25", "M2"),
    ("minilm", "M0"), ("minilm", "M1"), ("minilm", "M2"),
]


def load_export(retriever, memory):
    path = EXPORT_DIR / f"retrieval_{retriever}_{memory.lower()}.jsonl"
    records = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                row = json.loads(line)
                row["_line"] = line_number
                records.append(row)

    return records


def reply(record, style):
    return record.get(style, {}).get("candidate_text", "")


def score(record, style):
    value = record.get(style, {}).get("score")
    return float(value) if isinstance(value, (int, float)) else None


def is_fallback(record, style):
    item = record.get(style, {})
    return (
        item.get("source") == "fallback"
        or item.get("meta", {}).get("candidate_source") == "fallback"
    )


def pct(n, total):
    return round(100 * n / total, 1) if total else 0.0


def main():
    datasets = {
        config: load_export(*config)
        for config in CONFIGS
    }

    summaries = []
    low_overlap_examples = []

    print("RETRIEVAL EXPORT DIAGNOSTICS")
    print("=" * 78)

    for (retriever, memory), records in datasets.items():
        ids = [r.get("item_id") for r in records]
        duplicate_ids = len(ids) - len(set(ids))
        errors = sum(bool(r.get("errors")) for r in records)

        all_scores = []
        fallback_counts = Counter()
        reply_counts = {}
        unique_counts = {}

        for style in ("standard", "local"):
            texts = [reply(r, style).strip() for r in records]
            counts = Counter(texts)
            reply_counts[style] = counts
            unique_counts[style] = len(set(texts))
            fallback_counts[style] = sum(
                is_fallback(r, style) for r in records
            )

            for r in records:
                item = r.get(style, {})
                s = score(r, style)
                if s is not None:
                    all_scores.append(s)

                overlap = item.get("meta", {}).get("v6_overlap")
                if (
                    isinstance(s, (int, float))
                    and isinstance(overlap, (int, float))
                    and s >= 0.5
                    and overlap < 0.1
                ):
                    low_overlap_examples.append({
                        "item_id": r.get("item_id", ""),
                        "retriever": retriever,
                        "memory_mode": memory,
                        "style": style,
                        "message": r.get("message", ""),
                        "reply": item.get("candidate_text", ""),
                        "score": s,
                        "v6_overlap": overlap,
                        "retrieved_item_id": item.get("item_id", ""),
                    })

        mean_score = statistics.mean(all_scores) if all_scores else None

        summary = {
            "retriever": retriever,
            "memory_mode": memory,
            "rows": len(records),
            "duplicate_item_ids": duplicate_ids,
            "rows_with_errors": errors,
            "standard_fallbacks": fallback_counts["standard"],
            "local_fallbacks": fallback_counts["local"],
            "standard_unique_replies": unique_counts["standard"],
            "local_unique_replies": unique_counts["local"],
            "standard_most_common_reply_count": (
                reply_counts["standard"].most_common(1)[0][1]
                if records else 0
            ),
            "local_most_common_reply_count": (
                reply_counts["local"].most_common(1)[0][1]
                if records else 0
            ),
            "mean_final_score": round(mean_score, 4) if mean_score is not None else "",
            "min_final_score": round(min(all_scores), 4) if all_scores else "",
            "max_final_score": round(max(all_scores), 4) if all_scores else "",
        }
        summaries.append(summary)

        print(
            f"{retriever.upper():6} {memory}: "
            f"rows={len(records)}, errors={errors}, "
            f"fallbacks S/L={fallback_counts['standard']}/"
            f"{fallback_counts['local']}, "
            f"unique replies S/L={unique_counts['standard']}/"
            f"{unique_counts['local']}, "
            f"max repeated reply S/L="
            f"{summary['standard_most_common_reply_count']}/"
            f"{summary['local_most_common_reply_count']}, "
            f"mean score={summary['mean_final_score']}"
        )

    # Compare M1 and M2 outputs for the same retriever and query.
    print("\nM1 vs M2 COMPARISON")
    print("-" * 78)

    for retriever in ("bm25", "minilm"):
        m1 = {r.get("item_id"): r for r in datasets[(retriever, "M1")]}
        m2 = {r.get("item_id"): r for r in datasets[(retriever, "M2")]}
        common = sorted(set(m1) & set(m2))

        same_standard = sum(
            reply(m1[i], "standard") == reply(m2[i], "standard")
            for i in common
        )
        same_local = sum(
            reply(m1[i], "local") == reply(m2[i], "local")
            for i in common
        )
        same_both = sum(
            reply(m1[i], "standard") == reply(m2[i], "standard")
            and reply(m1[i], "local") == reply(m2[i], "local")
            for i in common
        )
        same_memory = sum(
            m1[i].get("memory", []) == m2[i].get("memory", [])
            for i in common
        )

        print(
            f"{retriever.upper()}: matched={len(common)}, "
            f"same STANDARD={same_standard}, same LOCAL={same_local}, "
            f"same both={same_both}, same memory={same_memory}"
        )

    summary_path = EXPORT_DIR / "retrieval_export_diagnostics.csv"
    with summary_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    examples_path = EXPORT_DIR / "high_score_low_overlap_cases.csv"
    with examples_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fields = [
            "item_id", "retriever", "memory_mode", "style",
            "message", "reply", "score", "v6_overlap",
            "retrieved_item_id",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(low_overlap_examples)

    print("\nSaved:", summary_path)
    print("Saved:", examples_path)
    print(
        "High-score/low-overlap cases:",
        len(low_overlap_examples),
    )
    print(
        "Note: overlap is a token-overlap diagnostic, "
        "not a semantic relevance metric."
    )


if __name__ == "__main__":
    main()
