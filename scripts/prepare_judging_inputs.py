"""Prepare anonymous development replies and a private system mapping; no model calls."""

import argparse
import json
from pathlib import Path

from replycoach.evaluation import DEFAULT_BANK, DEFAULT_QUERIES, validate_output_path
from replycoach.judging import SystemExport, prepare_judging_packet, write_judging_packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--system", action="append", nargs=3, required=True,
                        metavar=("LABEL", "EXPORT_JSONL", "MODE"),
                        help="Repeat for each system; MODE is retrieval_only, generator_only or hybrid")
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--items", type=int, help="Common pilot subset; default all validation items")
    parser.add_argument("--out-dir", type=Path, help="New directory outside data/; never reused")
    parser.add_argument("--check-only", action="store_true", help="Validate in memory without writing files")
    args = parser.parse_args()
    if not args.check_only and args.out_dir is None:
        parser.error("--out-dir is required unless --check-only is used")
    try:
        specs = [SystemExport(label, Path(path), mode) for label, path, mode in args.system]
        protected = (args.queries, args.bank, *(spec.path for spec in specs))
        if args.out_dir is not None:
            validate_output_path(args.out_dir, protected)
        public, private = prepare_judging_packet(args.queries, args.bank, specs,
                                                seed=args.seed, items=args.items)
        if not args.check_only:
            write_judging_packet(args.out_dir, public, private, protected_paths=protected)
        print(json.dumps({
            "status": "validated" if args.check_only else "prepared",
            "conversation_count": len({row["conversation_id"] for row in public}),
            "candidate_count": len(public), "system_count": len(specs),
            "output": str(args.out_dir) if args.out_dir is not None and not args.check_only else None,
            "note": "No LLM calls or ratings. Give judges only judge_inputs.jsonl, never PRIVATE_mapping.jsonl.",
        }))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Cannot prepare judging packet: {exc}\n"
                    "If an output directory was created before an I/O failure, it is incomplete; use a new directory.\n")


if __name__ == "__main__":
    main()
