"""Run known-answer training-critic diagnostics; never produce DPO labels."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from critic_preferences import PROMPT_VERSION, RUBRIC, SCHEMA, judge, save
from judge_replies import judge_settings, prepare_judge, read_env
from replycoach.io import ROOT, read_jsonl


def assess(first, second, expected, acceptable=None):
    """Check order consistency and, when supplied, both acceptance flags."""
    agreed = first["mapped_choice"] == second["mapped_choice"]
    correct = agreed and first["mapped_choice"] == expected
    if expected in ("first", "second") and (first["uncertain"] or second["uncertain"]):
        correct = False
    # Compare acceptability in original candidate order, independently of ranking.
    acceptance_errors = 0
    if acceptable is not None:
        for result, swapped in ((first, False), (second, True)):
            flags = [result["a_acceptable"], result["b_acceptable"]]
            if swapped:
                flags.reverse()
            acceptance_errors += sum(a != b for a, b in zip(flags, acceptable))
        correct = correct and acceptance_errors == 0
    return {"order_agreement": agreed, "correct": correct,
            **({"acceptance_errors": acceptance_errors} if acceptable is not None else {}),
            "unsafe_acceptance": any(
                any(flag and not allowed for flag, allowed in zip(
                    ([r["b_acceptable"], r["a_acceptable"]] if swapped else
                     [r["a_acceptable"], r["b_acceptable"]]), acceptable))
                for r, swapped in ((first, False), (second, True))) if acceptable is not None else
                expected == "both_bad" and any(
                    r["mapped_choice"] in ("first", "second") and not r["uncertain"] for r in (first, second))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new calibration output")
    cases = read_jsonl(args.cases)
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Nonempty unique calibration cases required")
    for c in cases:
        if c["expected"] not in ("first", "second", "tie", "both_bad"):
            raise ValueError("Invalid expected decision")
        if "acceptable" in c and (not isinstance(c["acceptable"], list)
                or len(c["acceptable"]) != 2
                or any(type(flag) is not bool for flag in c["acceptable"])):
            raise ValueError("Expected acceptability must contain exactly two booleans")
    config = judge_settings(read_env(args.env_file))
    settings = prepare_judge(config)
    report = {"purpose": "critic_calibration_only", "status": "running", "prompt_version": PROMPT_VERSION,
              "settings": settings, "rubric": RUBRIC, "schema": SCHEMA,
              "cases_sha256": sha256(args.cases.read_bytes()).hexdigest(),
              "limitation": "AI-authored diagnostic cases, not independent human gold", "results": []}
    save(args.output, report)
    try:
        for case in cases:
            # Only the conversation/style/replies reach the judge, never expected or case name.
            first, second = judge(case, config, settings), judge(case, config, settings, swap=True)
            result = {**case, "original": first, "swapped": second, **assess(first, second, case["expected"], case.get("acceptable"))}
            report["results"].append(result)
            save(args.output, report)
            print(json.dumps({k: result[k] for k in ("id", "correct", "unsafe_acceptance")}), flush=True)
    except Exception:
        report["status"] = "failed"
        save(args.output, report)
        raise
    report["status"] = "complete"
    report["summary"] = {"cases": len(cases), **dict(Counter({
        k: sum(r[k] for r in report["results"]) for k in ("correct", "order_agreement", "unsafe_acceptance")}))}
    save(args.output, report)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
