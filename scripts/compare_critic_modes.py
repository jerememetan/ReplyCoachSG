"""Compare pairwise flags with isolated-candidate verdicts; diagnostic output only."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from calibrate_critic import assess
from critic_preferences import PROMPT_VERSION, RUBRIC, SCHEMA, judge, save
from judge_replies import judge_settings, prepare_judge, read_env, request_json
from replycoach.io import ROOT, read_jsonl

SINGLE_VERSION = "selectable-single-v3-user-reviewed"
# Preserve the same task policy; remove only instructions about comparing candidates.
SINGLE_RUBRIC = RUBRIC.replace(
    "Compare A and B as selectable drafts of the USER's next reply to the incoming\nPEER message.",
    "Assess one selectable draft of the USER's next reply to the incoming PEER message.")
SINGLE_RUBRIC = SINGLE_RUBRIC.replace(
    "Do not infer a preference between equally useful\npersonal-status alternatives: choose tie when there is no clear quality difference.\n", "")
start = SINGLE_RUBRIC.index("Mark each candidate acceptable")
end = SINGLE_RUBRIC.index("STANDARD should use", start)
SINGLE_RUBRIC = (SINGLE_RUBRIC[:start] +
    "Wrong actors, contradictions, non-answers and incoherent replies are unacceptable.\n" +
    SINGLE_RUBRIC[end:]).replace(
    "Set uncertain true when you cannot reliably judge, including unfamiliar Singlish.",
    "Use verdict uncertain when you cannot reliably judge, including unfamiliar Singlish.")
SINGLE_RUBRIC += "\nExplain the draft's fit to the context, then return verdict acceptable, unacceptable or uncertain.\n"
SINGLE_SCHEMA = {"type": "object", "additionalProperties": False,
    "properties": {"reason": {"type": "string", "minLength": 1, "maxLength": 1000},
                   "verdict": {"type": "string", "enum": ["acceptable", "unacceptable", "uncertain"]}},
    "required": ["reason", "verdict"]}


def judge_single(case, reply, config, settings):
    # No second candidate, expected label, case name or ID reaches this request.
    task = {"conversation": case["conversation"], "incoming_peer_message": case["message"],
            "requested_style": case["style"], "draft": reply}
    response = request_json(config["base"], "/api/chat", {
        "model": settings["judge_model"], "stream": False, "think": settings["think"],
        "format": SINGLE_SCHEMA, "options": settings["options"],
        "messages": [{"role": "system", "content": SINGLE_RUBRIC},
                     {"role": "user", "content": json.dumps(task, ensure_ascii=False)}]})
    if response.get("done") is not True or response.get("done_reason") == "length":
        raise ValueError("Single-candidate response incomplete")
    result = json.loads(response["message"]["content"])
    if (not isinstance(result, dict) or set(result) != {"reason", "verdict"}
            or result["verdict"] not in ("acceptable", "unacceptable", "uncertain")
            or not isinstance(result["reason"], str) or not 1 <= len(result["reason"].strip()) <= 1000):
        raise ValueError("Invalid single-candidate verdict")
    return result


def score_single(results, expected):
    predictions = [r["verdict"] for r in results]
    return {"correct": all(p == ("acceptable" if e else "unacceptable")
                           for p, e in zip(predictions, expected)),
            "incorrect_acceptance": any(p == "acceptable" and not e for p, e in zip(predictions, expected)),
            "incorrect_rejection": any(p == "unacceptable" and e for p, e in zip(predictions, expected)),
            "uncertain_candidates": predictions.count("uncertain"),
            "candidate_correct": sum(p == ("acceptable" if e else "unacceptable")
                                     for p, e in zip(predictions, expected))}


def baseline_results(path, cases_path, cases, settings):
    baseline = json.loads(path.read_text())
    if (baseline.get("status") != "complete" or baseline.get("purpose") != "critic_calibration_only"
            or baseline.get("prompt_version") != PROMPT_VERSION
            or baseline.get("rubric") != RUBRIC or baseline.get("schema") != SCHEMA
            or baseline.get("cases_sha256") != sha256(cases_path.read_bytes()).hexdigest()
            or baseline.get("settings") != settings):
        raise ValueError("Baseline must match cases, rubric, model digest and runtime settings")
    rows = baseline.get("results", [])
    if len(rows) != len(cases) or any(any(row.get(k) != v for k, v in case.items())
                                    for row, case in zip(rows, cases)):
        raise ValueError("Baseline cases differ")
    return {r["id"]: r for r in rows}


def summarize(rows):
    return {"cases": len(rows), "single": {
        key: sum(r["single_metrics"][key] for r in rows)
        for key in ("correct", "incorrect_acceptance", "incorrect_rejection", "uncertain_candidates", "candidate_correct")},
        "pairwise": {key: sum(r["pairwise_metrics"][key] for r in rows)
                     for key in ("correct", "order_agreement", "unsafe_acceptance")},
        "pairwise_acceptance_correct": sum(r["pairwise_metrics"]["acceptance_errors"] == 0 for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh output")
    cases = read_jsonl(args.cases)
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Unique nonempty cases required")
    for c in cases:
        if (not isinstance(c.get("acceptable"), list) or len(c["acceptable"]) != 2
                or any(type(v) is not bool for v in c["acceptable"])
                or not isinstance(c.get("replies"), list) or len(c["replies"]) != 2
                or any(not isinstance(v, str) or not v.strip() for v in c["replies"])
                or c.get("expected") not in ("first", "second", "tie", "both_bad")):
            raise ValueError("Two replies, two expected acceptance flags and a valid preference required")
    config = judge_settings(read_env(args.env_file))
    settings = prepare_judge(config)
    baseline = baseline_results(args.baseline, args.cases, cases, settings) if args.baseline else {}
    report = {"purpose": "critic_mode_comparison_only", "status": "running", "settings": settings,
              "single_version": SINGLE_VERSION, "single_rubric": SINGLE_RUBRIC, "single_schema": SINGLE_SCHEMA,
              "pairwise_version": PROMPT_VERSION, "pairwise_rubric": RUBRIC, "pairwise_schema": SCHEMA,
              "cases_sha256": sha256(args.cases.read_bytes()).hexdigest(),
              "baseline_sha256": sha256(args.baseline.read_bytes()).hexdigest() if args.baseline else None,
              "limitation": "AI-authored diagnostics, not independent human gold; not preference labels",
              "results": []}
    save(args.output, report)
    try:
        for case in cases:
            if case["id"] in baseline:
                original, swapped = (baseline[case["id"]][k] for k in ("original", "swapped"))
            else:
                original, swapped = judge(case, config, settings), judge(case, config, settings, swap=True)
            isolated = [judge_single(case, reply, config, settings) for reply in case["replies"]]
            result = {**case, "pairwise_original": original, "pairwise_swapped": swapped,
                      "pairwise_metrics": assess(original, swapped, case["expected"], case["acceptable"]),
                      "single": isolated, "single_metrics": score_single(isolated, case["acceptable"])}
            report["results"].append(result)
            save(args.output, report)
            print(json.dumps({"id": case["id"], "single": result["single_metrics"],
                              "pairwise": result["pairwise_metrics"]}), flush=True)
    except Exception:
        report["status"] = "failed"
        save(args.output, report)
        raise
    report.update(status="complete", summary=summarize(report["results"]))
    save(args.output, report)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
