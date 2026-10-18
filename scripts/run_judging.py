"""One command: export development replies, anonymise a shared sample, judge it."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from judge_replies import ROOT, judge_settings, load_personas, prepare_judge, read_env


def run(script, *args, capture=False):
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / script), *map(str, args)],
        cwd=ROOT, check=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.PIPE if capture else None,
    )
    return json.loads(result.stdout) if capture else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=5, help="Shared validation conversations; default 5")
    parser.add_argument("--methods", nargs="+", choices=("bm25", "minilm"), default=["bm25"])
    parser.add_argument("--memories", nargs="+", choices=("M0", "M1", "M2"), default=["M0", "M2"])
    parser.add_argument("--personas", type=Path, default=ROOT / "data/evaluation/personas.csv")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--run-dir", type=Path, help="New run directory; default timestamped experiments/judging/run_...")
    parser.add_argument("--calibrate", action="store_true", help="Diagnostic run using the first persona only")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-only", action="store_true", help="Validate/preview; no judge requests or files")
    action.add_argument("--resume", action="store_true", help="Resume saved judging only; requires --run-dir")
    action.add_argument("--summary-only", action="store_true", help="Display saved results; requires --run-dir")
    args = parser.parse_args()
    if (args.resume or args.summary_only) and args.run_dir is None:
        parser.error("--resume/--summary-only requires --run-dir from the earlier run")
    if args.items < 1 or len(set(args.methods)) != len(args.methods) or len(set(args.memories)) != len(args.memories):
        parser.error("Use a positive item count and do not repeat methods/memory settings")
    out = (args.run_dir or ROOT / "experiments/judging" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    data = (ROOT / "data").resolve()
    if out == data or data in out.parents:
        raise ValueError("Keep judging outputs outside data/")
    display = out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else str(out)
    judge_args = [out / "packet", "--personas", args.personas, "--env-file", args.env_file]
    if args.calibrate:
        judge_args.append("--calibrate")
    if args.resume or args.summary_only:
        if not (out / "packet/judge_inputs.jsonl").exists():
            raise ValueError("This run did not reach judging. Preserve it and start a fresh pipeline run")
        run("judge_replies.py", *judge_args, "--resume" if args.resume else "--summary-only")
        return 0
    if out.exists():
        raise FileExistsError(f"Run directory already exists: {display}; use --resume or a new directory")

    print("[1/4] Checking development data, personas and judge setup...", flush=True)
    personas = load_personas(args.personas)
    if args.calibrate:
        personas = personas[:1]
    env_path = args.env_file
    if not env_path.exists():
        if env_path.resolve() != ROOT / ".env":
            raise FileNotFoundError(f"Custom environment file does not exist: {env_path}")
        env_path = ROOT / ".env.example"
    config = judge_settings(read_env(env_path))
    inputs = run("export_dev_outputs.py", "--check-only", capture=True)
    if args.items > inputs["queries"]:
        raise ValueError(f"--items must not exceed the {inputs['queries']} validation conversations")
    count = args.items * len(args.methods) * len(args.memories) * 2 * len(personas)
    print(f"Plan: {args.items} conversations x {len(args.methods) * len(args.memories)} settings x 2 styles "
          f"x {len(personas)} personas = {count} judgments. Judge: ollama/{config['model']}", flush=True)
    if args.calibrate:
        print("CALIBRATION: diagnostic only, not a validated evaluator or main result.", flush=True)
    if args.check_only:
        print("Validated. No judge requests or files created; installed local model readiness has not been checked.")
        return 0
    prepare_judge(config)
    if not args.env_file.exists():
        with args.env_file.open("x", encoding="utf-8") as handle:
            handle.write(env_path.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=False)
    print(f"Run folder: {display}", flush=True)
    resume = [".\\.venv\\Scripts\\python.exe", "scripts/run_judging.py", "--run-dir", display, "--resume"]
    if args.calibrate:
        resume.append("--calibrate")
    if args.personas != ROOT / "data/evaluation/personas.csv":
        resume.extend(["--personas", str(args.personas)])
    if args.env_file != ROOT / ".env":
        resume.extend(["--env-file", str(args.env_file)])
    print("If interrupted while judging, resume with:\n" + subprocess.list2cmdline(resume), flush=True)

    print("[2/4] Generating the compared model replies...", flush=True)
    systems = []
    for method in args.methods:
        for memory in args.memories:
            label = f"{'BM25' if method == 'bm25' else 'MiniLM'}_{memory}"
            export = out / "exports" / f"{label}.jsonl"
            run("export_dev_outputs.py", "--mode", "retrieval_only", "--retrieval", method,
                "--memory", memory, "--output", export)
            systems.extend(["--system", label, export, "retrieval_only"])
    print("[3/4] Preparing the same anonymous conversations for every setting...", flush=True)
    run("prepare_judging_inputs.py", *systems, "--items", args.items, "--out-dir", out / "packet")
    print("[4/4] Running persona judgments and showing model-setting summaries...", flush=True)
    run("judge_replies.py", *judge_args)
    print(f"Results: {display}/packet/ratings/summary.csv", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped. Completed outputs are preserved; use the printed resume command if judging started.", file=sys.stderr)
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        print("Pipeline stopped at a failed stage. Existing outputs are preserved; see the error above.", file=sys.stderr)
        raise SystemExit(exc.returncode or 1)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Cannot run judging pipeline: {exc}", file=sys.stderr)
        raise SystemExit(2)
