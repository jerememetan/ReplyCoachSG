
"""Benchmark retrieval-only latency across BM25/MiniLM and M0/M1/M2.

Uses the 50 validation queries and the existing 190-item training bank.
Measures retrieval + ranking time, excluding retriever initialisation.
Does not modify existing exports or access the held-out test set.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from replycoach.evaluation import (
    DEFAULT_BANK,
    DEFAULT_QUERIES,
    load_evaluation_inputs,
)
from replycoach.systems import run_system
from replycoach.retrieval.v0 import V0Retriever


MEMORY_MODES = ("M0", "M1", "M2")
RETRIEVAL_METHODS = ("bm25", "minilm")


def percentile(values: list[float], p: float) -> float:
    """Return a nearest-rank percentile."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((p * len(ordered) + 0.999999)) - 1))
    return ordered[index]


def is_fallback(candidate) -> bool:
    return (
        candidate.source == "fallback"
        or candidate.meta.get("reason") == "low_confidence_or_unknown"
    )


def build_retriever(method: str, bank_items):
    """Build each retrieval index once, outside the timed query loop."""
    if method == "bm25":
        return V0Retriever.from_items(bank_items)

    from replycoach.retrieval.v2 import V2Retriever

    return V2Retriever.from_items(bank_items, cache_file=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark BM25/MiniLM retrieval latency across M0/M1/M2."
    )
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--bank", default=str(DEFAULT_BANK))
    parser.add_argument(
        "--output-dir",
        default="experiments",
        help="Directory for the benchmark CSV files.",
    )
    args = parser.parse_args()

    inputs = load_evaluation_inputs(args.queries, args.bank)
    queries = inputs.queries.items
    bank_items = inputs.bank.items

    if not queries:
        raise ValueError("No validation queries were loaded.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    summary_rows = []

    print(f"Validation queries: {len(queries)}")
    print(f"Training bank items: {len(bank_items)}")
    print("Timing excludes retriever construction and one warm-up query.")
    print()

    for method in RETRIEVAL_METHODS:
        print(f"Initialising {method.upper()} retriever...")
        retriever = build_retriever(method, bank_items)

        # Reuse this retriever instance for every query and memory mode.
        retriever_factory = lambda retriever=retriever: retriever

        for memory_mode in MEMORY_MODES:
            print(f"Benchmarking {method.upper()} / {memory_mode}...")

            # Warm up without including this run in the measurements.
            warmup_item = queries[0]
            run_system(
                warmup_item.message.text,
                warmup_item.memory_texts(),
                memory_mode,
                mode="retrieval_only",
                retrieval_method=method,
                retriever_factory=retriever_factory,
                generator_factory=None,
                log=False,
            )

            latencies_ms = []
            standard_fallbacks = 0
            local_fallbacks = 0
            errors = 0

            for item in queries:
                start = time.perf_counter()

                output = run_system(
                    item.message.text,
                    item.memory_texts(),
                    memory_mode,
                    mode="retrieval_only",
                    retrieval_method=method,
                    retriever_factory=retriever_factory,
                    generator_factory=None,
                    log=False,
                )

                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies_ms.append(elapsed_ms)

                standard = output.result.standard
                local = output.result.local

                standard_fallback = is_fallback(standard)
                local_fallback = is_fallback(local)

                standard_fallbacks += int(standard_fallback)
                local_fallbacks += int(local_fallback)
                errors += int(bool(output.errors))

                detail_rows.append(
                    {
                        "retrieval_method": method,
                        "memory_mode": memory_mode,
                        "item_id": item.id,
                        "latency_ms": round(elapsed_ms, 4),
                        "standard_fallback": standard_fallback,
                        "local_fallback": local_fallback,
                        "has_errors": bool(output.errors),
                    }
                )

            summary_rows.append(
                {
                    "retrieval_method": method,
                    "memory_mode": memory_mode,
                    "query_count": len(queries),
                    "mean_latency_ms": round(statistics.mean(latencies_ms), 4),
                    "median_latency_ms": round(statistics.median(latencies_ms), 4),
                    "p95_latency_ms": round(percentile(latencies_ms, 0.95), 4),
                    "standard_fallbacks": standard_fallbacks,
                    "local_fallbacks": local_fallbacks,
                    "queries_with_errors": errors,
                }
            )

    detail_path = output_dir / "retrieval_latency_per_query.csv"
    summary_path = output_dir / "retrieval_latency_summary.csv"

    with detail_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
        writer.writeheader()
        writer.writerows(detail_rows)

    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nBenchmark complete.")
    print(f"Per-query results: {detail_path}")
    print(f"Summary results:   {summary_path}")
    print("\nSummary:")
    for row in summary_rows:
        print(
            f"{row['retrieval_method'].upper():6} {row['memory_mode']} | "
            f"mean={row['mean_latency_ms']:.2f} ms | "
            f"median={row['median_latency_ms']:.2f} ms | "
            f"p95={row['p95_latency_ms']:.2f} ms | "
            f"fallbacks S/L="
            f"{row['standard_fallbacks']}/{row['local_fallbacks']} | "
            f"errors={row['queries_with_errors']}"
        )


if __name__ == "__main__":
    main()
