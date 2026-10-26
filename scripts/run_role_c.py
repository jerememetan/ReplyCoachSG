"""Reproduce Role C's full CPU/GPU experiment without reading held-out test data.

Example: python scripts/run_role_c.py --run-dir artifacts/role_c_runs/my_run
Uses local DailyDialog files; no downloads, hosted calls or generated gold data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

import torch

from replycoach.evaluation import DEFAULT_BANK, DEFAULT_QUERIES, load_evaluation_inputs
from replycoach.generate.rnn_provider import RNNGenerator, file_hash
from replycoach.generate.train_v3 import run_one_batch, train_dailydialog
from replycoach.pipeline import prepare_input
from replycoach.tokenize import tokenize


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def code_hashes():
    paths = [*sorted((ROOT / "src").rglob("*.py")), ROOT / "scripts/run_role_c.py",
             ROOT / "scripts/prepare_dailydialog_pairs.py"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def audit(run_dir):
    """Inspect only dev/public inputs and old RNN artifacts, never test rows."""
    inputs = load_evaluation_inputs(DEFAULT_QUERIES, DEFAULT_BANK)
    manifest = json.loads((ROOT / "data/split/manifest.json").read_text(encoding="utf-8"))
    for label, items in (("train", inputs.bank.items), ("validation", inputs.queries.items)):
        if any(manifest["assignment"].get(i.id) != label for i in items):
            raise ValueError("development files do not match the frozen split manifest")
    groups = manifest["group_of"]
    if {groups[i.id] for i in inputs.bank.items if groups[i.id] is not None} & {groups[i.id] for i in inputs.queries.items if groups[i.id] is not None}:
        raise ValueError("near-duplicate families cross the train/validation split")
    result = {"train_items": len(inputs.bank.items), "validation_items": len(inputs.queries.items),
              "held_out_test_read": False, "public": {}, "historical_checkpoints": {}}
    dialogue_sets = []
    for split in ("train", "validation"):
        raw = ROOT / f"data/public/dailydialog_{split}.jsonl"
        original = ROOT / f"data/public/dailydialog_{split}_pairs.jsonl"
        regenerated = run_dir / f"dailydialog_{split}_pairs.jsonl"
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_dailydialog_pairs.py"),
                        "--input", str(raw), "--output", str(regenerated)], check=True)
        with original.open(encoding="utf-8") as old, regenerated.open(encoding="utf-8") as new:
            count = 0
            for left, right in itertools.zip_longest(old, new):
                if left is None or right is None or json.loads(left) != json.loads(right):
                    raise ValueError(f"{split}: existing pairs differ from exact preparation script")
                count += 1
        with raw.open(encoding="utf-8") as handle:
            dialogues = {tuple(" ".join(tokenize(t)) for t in json.loads(line)["utterances"])
                         for line in handle if line.strip()}
        dialogue_sets.append(dialogues)
        result["public"][split] = {"pairs": count, "raw_sha256": file_hash(raw),
                                     "regenerated_sha256": file_hash(regenerated),
                                     "original_sha256": file_hash(original)}
    result["public"]["shared_complete_dialogues"] = len(dialogue_sets[0] & dialogue_sets[1])
    if result["public"]["shared_complete_dialogues"]:
        regenerated = run_dir / "dailydialog_train_pairs.jsonl"
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_dailydialog_pairs.py"),
                        "--input", str(ROOT / "data/public/dailydialog_train.jsonl"),
                        "--exclude-dialogues", str(ROOT / "data/public/dailydialog_validation.jsonl"),
                        "--output", str(regenerated)], check=True)
        with regenerated.open(encoding="utf-8") as handle:
            result["public"]["train"]["clean_pairs"] = sum(bool(line.strip()) for line in handle)
        result["public"]["train"]["clean_sha256"] = file_hash(regenerated)
        result["public"]["policy"] = "Remove complete training dialogues matching validation after tokenizer normalization; originals unchanged."
    for version in ("v3", "v4"):
        path = ROOT / f"artifacts/rnn_{version}_dailydialog.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            result["historical_checkpoints"][version] = {
                "sha256": file_hash(path), "epoch": saved.get("epoch"),
                "validation_loss": saved.get("validation_loss"),
                "use_attention": saved.get("use_attention")}
    write_json(run_dir / "audit.json", result)
    return result


def token_f1(prediction, reference):
    pred, gold = Counter(tokenize(prediction)), Counter(tokenize(reference))
    overlap = sum((pred & gold).values())
    return 2 * overlap / (sum(pred.values()) + sum(gold.values())) if pred or gold else 1.0


def compare_task(run_dir):
    inputs = load_evaluation_inputs(DEFAULT_QUERIES, DEFAULT_BANK)
    summary = {"evaluation_set": "50 shared development validation items; M2 memory",
               "metrics_limitations": "Token F1/exact match measure reference overlap, not appropriateness or style.",
               "v4_vs_v5": {"status": "pending_role_d_trained_provider",
                            "winner": None, "reason": "V5 is not task-trained/connected; no scores fabricated."}}
    for version in ("v3", "v4"):
        generator = RNNGenerator(run_dir / f"{version}_task.pt")
        rows, scores, elapsed = [], {style: [] for style in ("STANDARD", "LOCAL")}, []
        for item in inputs.queries.items:
            memory = prepare_input(item.message.text, item.memory_texts(), "M2").memory
            started = time.perf_counter()
            candidates = generator.generate(item.message.text, memory)
            elapsed.append(time.perf_counter() - started)
            for candidate, reference in zip(candidates, (item.replies.standard, item.replies.local)):
                f1 = token_f1(candidate.candidate_text, reference)
                scores[candidate.style].append(f1)
                rows.append({"item_id": item.id, "reference": reference,
                             "token_f1": f1, "candidate": candidate.to_dict()})
        with (run_dir / f"{version}_validation.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        report = json.loads((run_dir / f"{version}_task.json").read_text(encoding="utf-8"))
        summary[version] = {"best_epoch": report["best_epoch"],
                            "validation_loss": report["best_validation_loss"],
                            "mean_pair_latency_seconds": sum(elapsed) / len(elapsed),
                            "token_f1": {s: sum(v) / len(v) for s, v in scores.items()},
                            "exact_matches": sum(tokenize(r["candidate"]["candidate_text"]) == tokenize(r["reference"]) for r in rows),
                            "style_pairs_with_different_text": sum(rows[i]["candidate"]["candidate_text"] != rows[i+1]["candidate"]["candidate_text"] for i in range(0, len(rows), 2)),
                            "outputs": len(rows), "checkpoint_sha256": generator.checkpoint_hash}
    summary["rnn_minimum_validation_loss"] = min(("v3", "v4"), key=lambda v: summary[v]["validation_loss"])
    summary["deployment_note"] = "V4 supplied for the agreed V4/V5 integration; lower loss alone does not establish quality."
    write_json(run_dir / "comparison.json", summary)
    return summary


def import_pretrained(run_dir, v3_path, v4_path):
    """Reuse verified complete historical runs; never silently resume/retrain them."""
    imported = {"mode": "reuse_existing_five_epoch_pretraining",
                "known_limitation": "Historical DailyDialog splits contain shared complete dialogues. Clean pairs in this run were prepared for future reruns, not used to train imported weights.",
                "versions": {}}
    for version, path in (("v3", v3_path), ("v4", v4_path)):
        report_path = path.with_suffix(".json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("epoch") != report["best_epoch"] or bool(saved.get("use_attention")) != (version == "v4"):
            raise ValueError("imported pretraining architecture/epoch does not match report")
        if round(float(saved["validation_loss"]), 4) != report["best_validation_loss"]:
            raise ValueError("imported checkpoint validation loss does not match report")
        if report["train_pairs"] != 76052 or report["validation_pairs"] != 7069 or len(report["history"]) != 5:
            raise ValueError("expected a complete five-epoch historical DailyDialog run")
        shutil.copy2(path, run_dir / f"{version}_public.pt")
        shutil.copy2(report_path, run_dir / f"{version}_public.json")
        imported["versions"][version] = {"source": str(path.resolve()), "checkpoint_sha256": file_hash(path),
                                           "report_sha256": file_hash(report_path), "epoch": saved["epoch"],
                                           "saved_validation_loss": saved["validation_loss"]}
        print(f"REUSED {version} epoch {saved['epoch']} checkpoint (no pretraining)", flush=True)
    write_json(run_dir / "imported_pretraining.json", imported)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="New directory under artifacts/role_c_runs")
    parser.add_argument("--stage", choices=("all", "pretrain", "task"), default="all")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--task-epochs", type=int, default=60)
    parser.add_argument("--reuse-pretrained", action="store_true", help="Reuse existing complete V3/Colab V4 runs instead of repeating DailyDialog training")
    parser.add_argument("--v3-pretrained", type=Path, default=ROOT / "artifacts/rnn_v3_dailydialog.pt")
    parser.add_argument("--v4-pretrained", type=Path, default=ROOT / "artifacts/colab_import/rnn_v4_dailydialog.pt")
    args = parser.parse_args()
    target = args.run_dir.resolve()
    allowed = (ROOT / "artifacts/role_c_runs").resolve()
    if allowed not in target.parents:
        parser.error("--run-dir must be a child of artifacts/role_c_runs")
    if min(args.threads, args.pretrain_epochs, args.task_epochs) < 1:
        parser.error("threads and epochs must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.stage in ("all", "pretrain"):
        target.mkdir(parents=True, exist_ok=False)
        environment = {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(),
                       "cuda": torch.cuda.is_available(), "threads": torch.get_num_threads(), "seed": 425,
                       "deterministic_algorithms": True, "command": sys.argv, "code_sha256": code_hashes()}
        write_json(target / "environment.json", environment)
        with (target / "requirements_frozen.txt").open("w", encoding="utf-8") as frozen:
            subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze", "--exclude-editable"], stdout=frozen, check=True)
        audit(target)
        smoke = run_one_batch(output_dir=target / "smoke")
        print(json.dumps({"stage": "smoke", **smoke}), flush=True)
        if args.reuse_pretrained:
            import_pretrained(target, args.v3_pretrained, args.v4_pretrained)
        else:
            for attention in (False, True):
                version = "v4" if attention else "v3"
                print(f"START {version} full DailyDialog pretraining", flush=True)
                train_dailydialog(target / "dailydialog_train_pairs.jsonl", target / "dailydialog_validation_pairs.jsonl",
                                 epochs=args.pretrain_epochs, output_name=f"{version}_public", output_dir=target,
                                 use_attention=attention)
    if args.stage in ("all", "task"):
        from replycoach.generate.train_task import train_task

        if (target / "comparison.json").exists():
            raise FileExistsError("comparison exists; use a fresh run")
        for version in ("v3", "v4"):
            print(f"START {version} task fine-tuning", flush=True)
            train_task(target / f"{version}_public.pt", DEFAULT_BANK, DEFAULT_QUERIES, target,
                       epochs=args.task_epochs)
        comparison = compare_task(target)
        os.environ["REPLYCOACH_V4_CHECKPOINT"] = str(target / "v4_task.pt")
        subprocess.run([sys.executable, str(ROOT / "scripts/export_dev_outputs.py"), "--generator", "v4",
                        "--mode", "all", "--output", str(target / "three_modes_v4.jsonl")], check=True)
        write_json(target / "completion.json", {"role_c_training_and_export_completed": True,
                   "held_out_test_read": False, "v4_v5_selection_completed": False,
                   "final_code_sha256": code_hashes(), "comparison": comparison})
        print(json.dumps(comparison, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
