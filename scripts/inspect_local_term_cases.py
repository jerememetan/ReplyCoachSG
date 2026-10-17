"""Inspect local-term preservation and additions in development exports.

Read-only diagnostic. Does not modify data, models, or retrieval behaviour.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEXICON_PATH = ROOT / "data" / "lexicons" / "local_terms.json"
EXPORT_DIR = ROOT / "experiments"


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def contains_surface(text, surface):
    text = text.casefold()
    surface = surface.casefold()

    # Match Chinese characters as substrings.
    if any("\u3400" <= char <= "\u9fff" for char in surface):
        return surface in text

    # Match Latin terms using word boundaries.
    pattern = (
        r"(?<![A-Za-z0-9_])"
        + re.escape(surface)
        + r"(?![A-Za-z0-9_])"
    )
    return re.search(pattern, text) is not None


def matched_terms(text, terms):
    return {
        term["surface"]
        for term in terms
        if term.get("apply", True)
        and contains_surface(text, term["surface"])
    }


def show_text(label, text):
    print(f"{label}:")
    print(f"  {text if text else '(empty)'}")


def main():
    lexicon = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    terms = lexicon["terms"]

    eligible = [
        term for term in terms
        if term.get("apply", True)
    ]

    # Exclude particles, Mandarin, and pinyin from core-term analysis.
    core = [
        term for term in eligible
        if term.get("kind") not in {"particle", "zh", "pinyin"}
    ]

    exports = sorted(EXPORT_DIR.glob("retrieval_*.jsonl"))

    if not exports:
        raise SystemExit(
            "No experiments/retrieval_*.jsonl files found."
        )

    for path in exports:
        rows = load_jsonl(path)

        print("\n" + "=" * 78)
        print(f"EXPORT: {path.name}")
        print("=" * 78)

        case_count = 0

        for row in rows:
            message = row["message"]
            memory_items = row.get("memory", [])
            memory = " ".join(memory_items)

            local_text = row["local"].get("candidate_text", "")
            standard_text = row["standard"].get("candidate_text", "")

            source_terms = matched_terms(message, core)
            local_terms = matched_terms(local_text, core)
            grounded_terms = matched_terms(
                f"{message} {memory}", core
            )

            # Terms present in the message but absent from LOCAL reply.
            missing_terms = source_terms - local_terms

            # Terms in LOCAL reply that aren't grounded in message/memory.
            unprompted_terms = local_terms - grounded_terms

            # Only show cases that warrant inspection.
            if not source_terms and not unprompted_terms:
                continue

            case_count += 1

            print(f"\nCase {case_count}: {row['item_id']}")
            print(f"Memory mode: {row.get('memory_mode')}")
            print(f"Retriever: {row.get('retrieval_method')}")
            print(f"Source terms: {sorted(source_terms)}")
            print(f"Missing source terms: {sorted(missing_terms)}")
            print(f"Unprompted core additions: {sorted(unprompted_terms)}")

            show_text("Original message", message)

            if memory_items:
                print("Selected memory:")
                for index, turn in enumerate(memory_items, start=1):
                    print(f"  {index}. {turn}")
            else:
                print("Selected memory: (none)")

            show_text("STANDARD reply", standard_text)
            print(
                "STANDARD source:",
                row["standard"].get("source"),
            )
            show_text("LOCAL reply", local_text)
            print("LOCAL source:", row["local"].get("source"))

        if case_count == 0:
            print("No cases matched the inspection criteria.")

        print(f"\nCases printed: {case_count} of {len(rows)} rows")


if __name__ == "__main__":
    main()
