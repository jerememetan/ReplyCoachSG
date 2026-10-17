
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

RETRIEVERS = ("bm25", "minilm")
MEMORY_MODES = ("M0", "M1", "M2")


def read_jsonl(path):
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row["item_id"]
            if item_id in rows:
                raise ValueError(f"Duplicate item {item_id} in {path.name}")
            rows[item_id] = row
    return rows


def reply(row, style):
    candidate = row.get(style.lower()) or {}
    return candidate.get("candidate_text", "")


def candidate_field(row, style, field):
    candidate = row.get(style.lower()) or {}
    value = candidate.get(field, "")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def read_human_ratings():
    path = EXPERIMENTS / "manual_relevance_scoring.csv"
    ratings = {}

    if not path.exists():
        print("Warning: manual ratings file not found; continuing without ratings.")
        return ratings

    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["item_id"].strip(),
                row["retriever"].strip().lower(),
                row["memory_mode"].strip().upper(),
            )
            ratings[key] = row

    return ratings


def main():
    exports = {}
    for retriever in RETRIEVERS:
        for memory in MEMORY_MODES:
            key = (retriever, memory)
            path = EXPERIMENTS / f"retrieval_{retriever}_{memory.lower()}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing export: {path}")
            exports[key] = read_jsonl(path)

    item_sets = [set(rows) for rows in exports.values()]
    if not all(ids == item_sets[0] for ids in item_sets):
        raise ValueError("The six exports do not contain matching query IDs.")

    ratings = read_human_ratings()
    results = []

    for item_id in sorted(item_sets[0]):
        base = exports[("bm25", "M0")][item_id]
        out = {
            "item_id": item_id,
            "message": base.get("message", ""),
            "memory_m0": json.dumps(base.get("memory", []), ensure_ascii=False),
        }

        standard_replies = {}
        local_replies = {}
        error_configs = []

        for retriever in RETRIEVERS:
            for memory in MEMORY_MODES:
                key = (retriever, memory)
                row = exports[key][item_id]
                prefix = f"{retriever}_{memory.lower()}"

                standard_replies[key] = reply(row, "standard")
                local_replies[key] = reply(row, "local")

                out[f"{prefix}_standard"] = standard_replies[key]
                out[f"{prefix}_local"] = local_replies[key]
                out[f"{prefix}_standard_score"] = candidate_field(
                    row, "standard", "score"
                )
                out[f"{prefix}_local_score"] = candidate_field(
                    row, "local", "score"
                )
                out[f"{prefix}_standard_source"] = candidate_field(
                    row, "standard", "source"
                )
                out[f"{prefix}_local_source"] = candidate_field(
                    row, "local", "source"
                )
                out[f"{prefix}_retrieval_count"] = row.get("retrieval_count", "")
                out[f"{prefix}_errors"] = json.dumps(
                    row.get("errors", {}), ensure_ascii=False
                )

                if row.get("errors"):
                    error_configs.append(prefix)

                human = ratings.get((item_id, retriever, memory), {})
                for style in ("standard", "local"):
                    score_col = f"{style}_relevance_0_2"
                    out[f"{prefix}_{style}_human_score"] = human.get(
                        score_col, ""
                    )
                out[f"{prefix}_coherence_issue"] = human.get(
                    "coherence_issue", ""
                )

        # Does changing the retriever change the reply at the same memory mode?
        retriever_diffs = []
        for memory in MEMORY_MODES:
            for style, reply_map in (
                ("standard", standard_replies),
                ("local", local_replies),
            ):
                if reply_map[("bm25", memory)] != reply_map[("minilm", memory)]:
                    retriever_diffs.append(f"{memory}_{style}")

        # Does adding/changing memory change the reply for the same retriever?
        memory_diffs = []
        for retriever in RETRIEVERS:
            for memory in ("M1", "M2"):
                for style, reply_map in (
                    ("standard", standard_replies),
                    ("local", local_replies),
                ):
                    if reply_map[(retriever, "M0")] != reply_map[(retriever, memory)]:
                        memory_diffs.append(f"{retriever}_{memory}_{style}")

        m1_m2_diffs = []
        for retriever in RETRIEVERS:
            for style, reply_map in (
                ("standard", standard_replies),
                ("local", local_replies),
            ):
                if reply_map[(retriever, "M1")] != reply_map[(retriever, "M2")]:
                    m1_m2_diffs.append(f"{retriever}_{style}")

        # Human flags are only available for the subset already rated.
        human_flags = []
        for retriever in RETRIEVERS:
            for memory in MEMORY_MODES:
                human = ratings.get((item_id, retriever, memory), {})
                scores = []
                for col in (
                    "standard_relevance_0_2",
                    "local_relevance_0_2",
                ):
                    try:
                        scores.append(int(human.get(col, "").strip()))
                    except (ValueError, AttributeError):
                        pass

                if scores and min(scores) < 2:
                    human_flags.append(f"{retriever}_{memory}_low_score")

                if human.get("coherence_issue", "").strip().lower() == "yes":
                    human_flags.append(f"{retriever}_{memory}_coherence")

        out["retriever_disagreements"] = ";".join(retriever_diffs)
        out["memory_changes_vs_m0"] = ";".join(memory_diffs)
        out["m1_vs_m2_differences"] = ";".join(m1_m2_diffs)
        out["human_review_flags"] = ";".join(sorted(set(human_flags)))
        out["error_configs"] = ";".join(error_configs)

        out["review_priority"] = (
            len(retriever_diffs)
            + len(memory_diffs)
            + len(human_flags)
            + 3 * len(error_configs)
        )

        results.append(out)

    results.sort(key=lambda row: (-row["review_priority"], row["item_id"]))

    full_path = EXPERIMENTS / "configuration_comparison.csv"
    queue_path = EXPERIMENTS / "configuration_review_queue.csv"

    columns = list(results[0].keys()) if results else ["item_id"]
    with full_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    # Include queries with a disagreement, a human flag, or an export error.
    queue = [
        row for row in results
        if row["retriever_disagreements"]
        or row["memory_changes_vs_m0"]
        or row["human_review_flags"]
        or row["error_configs"]
    ]

    with queue_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(queue)

    print("CONFIGURATION COMPARISON")
    print("=" * 60)
    print("Validation queries compared:", len(results))
    print("Queries with retriever disagreements:",
          sum(bool(r["retriever_disagreements"]) for r in results))
    print("Queries where memory changes a reply:",
          sum(bool(r["memory_changes_vs_m0"]) for r in results))
    print("Queries where M1 and M2 differ:",
          sum(bool(r["m1_vs_m2_differences"]) for r in results))
    print("Queries with existing human-rating flags:",
          sum(bool(r["human_review_flags"]) for r in results))
    print("Queries with export errors:",
          sum(bool(r["error_configs"]) for r in results))
    print("Review queue size:", len(queue))
    print("Saved:", full_path)
    print("Saved:", queue_path)
    print("Note: disagreement and priority flags do not establish relevance.")


if __name__ == "__main__":
    main()
