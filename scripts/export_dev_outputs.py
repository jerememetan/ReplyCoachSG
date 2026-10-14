"""Export matched development outputs, never held-out test or partial hybrids."""

import argparse
import json
from pathlib import Path

from replycoach.evaluation import (
    DEFAULT_BANK, DEFAULT_QUERIES, collect_dev_outputs, load_evaluation_inputs,
    validate_output_path, write_dev_outputs,
)
from replycoach.generate.providers import load_generator
from replycoach.retrieval.v0 import V0Retriever
from replycoach.systems import SYSTEM_MODES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="Reviewed dev validation JSONL")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK, help="Disjoint reviewed dev reply bank")
    parser.add_argument("--generator", choices=("v4", "v5"), default="v4")
    parser.add_argument("--retrieval", choices=("bm25", "minilm"), default="bm25")
    parser.add_argument("--memory", choices=("M0", "M1", "M2"), default="M2")
    parser.add_argument("--mode", choices=(*SYSTEM_MODES, "all"), default="all",
                        help="Default all compares the three modes; retrieval_only needs no generator")
    parser.add_argument("--output", type=Path, help="New JSONL filename outside data/")
    parser.add_argument("--check-only", action="store_true", help="Validate input files without loading any model or writing output")
    args = parser.parse_args()
    if not args.check_only and args.output is None:
        parser.error("--output is required unless --check-only is used")
    try:
        inputs = load_evaluation_inputs(args.queries, args.bank)
        if args.check_only:
            print(json.dumps({"status": "data_valid", "queries": len(inputs.queries.items),
                              "bank_items": len(inputs.bank.items),
                              "note": "Models have not been loaded; this is not an evaluation result."}))
            return
        validate_output_path(args.output, (args.queries, args.bank))
        if args.mode == "generator_only":
            retriever = None
        elif args.retrieval == "bm25":
            retriever = V0Retriever.from_items(inputs.bank.items)
        else:
            from replycoach.retrieval.v2 import V2Retriever

            retriever = V2Retriever.from_items(inputs.bank.items, cache_file=None)
        generator = None if args.mode == "retrieval_only" else load_generator(args.generator)
        rows = collect_dev_outputs(
            inputs, retriever_factory=lambda: retriever, generator_factory=lambda: generator,
            retrieval_method=args.retrieval, memory_mode=args.memory, mode=args.mode,
        )
        write_dev_outputs(args.output, rows, protected_paths=(args.queries, args.bank))
        print(json.dumps({"status": "exported", "rows": len(rows), "output": str(args.output)}))
    except FileNotFoundError as exc:
        parser.exit(2, f"Role A: development files are not ready. See docs/role_handoffs.md. {exc}\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Cannot export development comparison: {exc}\n")


if __name__ == "__main__":
    main()
