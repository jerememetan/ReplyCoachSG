"""Reload actual checkpoints and verify Role C results; no test-set access."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch import nn
from torch.utils.data import DataLoader

from replycoach.evaluation import DEFAULT_BANK, DEFAULT_QUERIES, load_evaluation_inputs
from replycoach.generate.rnn_provider import RNNGenerator, file_hash
from replycoach.generate.train_task import task_pairs
from replycoach.generate.train_v3 import collate, encode_pair, epoch_loss, load_public_pairs
from replycoach.generate import Vocab
from replycoach.generate.seq2seq import Seq2Seq
from replycoach.pipeline import prepare_input
from replycoach.tokenize import tokenize


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(run_dir):
    torch.set_num_threads(2)
    inputs = load_evaluation_inputs(DEFAULT_QUERIES, DEFAULT_BANK)
    by_id = {item.id: item for item in inputs.queries.items}
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    results = {"held_out_test_read": False, "versions": {}}
    comparison = read_json(run_dir / "comparison.json")
    for version in ("v3", "v4"):
        public = read_json(run_dir / f"{version}_public.json")
        saved = torch.load(run_dir / f"{version}_public.pt", map_location="cpu", weights_only=True)
        vocab = Vocab.from_dict(saved["vocab"])
        model = Seq2Seq(len(vocab), saved["embed_dim"], saved["hidden_size"], saved["use_attention"])
        model.load_state_dict(saved["model_state"])
        pairs = load_public_pairs(run_dir / "dailydialog_validation_pairs.jsonl")
        encoded = [encode_pair(pair, vocab, saved["max_source_tokens"], saved["max_target_tokens"]) for pair in pairs]
        value = epoch_loss(model, DataLoader(encoded, batch_size=32, collate_fn=collate), criterion, torch.device("cpu"))
        if abs(value - saved["validation_loss"]) > 1e-5 or round(value, 4) != public["best_validation_loss"]:
            raise ValueError(f"{version}: public validation does not match checkpoint/report")
        generator = RNNGenerator(run_dir / f"{version}_task.pt")
        task = read_json(run_dir / f"{version}_task.json")
        for key, path in (("train_sha256", DEFAULT_BANK), ("validation_sha256", DEFAULT_QUERIES)):
            if generator.provenance[key] != file_hash(path):
                raise ValueError("task input files changed since training")
        if generator.provenance["pretrained_sha256"] != file_hash(run_dir / f"{version}_public.pt"):
            raise ValueError("task checkpoint is linked to different pretrained weights")
        encoded = task_pairs(inputs.queries.items, generator.vocab, generator.max_source, generator.max_target)
        value_task = epoch_loss(generator.model, DataLoader(encoded, batch_size=32, collate_fn=collate),
                                criterion, torch.device("cpu"))
        if abs(value_task - task["best_validation_loss"]) > 1e-5:
            raise ValueError(f"{version}: task validation loss mismatch")
        if generator.checkpoint_hash != comparison[version]["checkpoint_sha256"] or generator.checkpoint_hash != task["checkpoint_sha256"]:
            raise ValueError(f"{version}: checkpoint fingerprints disagree")
        rows = [json.loads(line) for line in (run_dir / f"{version}_validation.jsonl").read_text(encoding="utf-8").splitlines()]
        keys = {(row["item_id"], row["candidate"]["style"]) for row in rows}
        if len(rows) != 100 or keys != {(item.id, style) for item in inputs.queries.items for style in ("STANDARD", "LOCAL")}:
            raise ValueError("validation generation coverage mismatch")
        regenerated = {}
        for item in inputs.queries.items:
            memory = prepare_input(item.message.text, item.memory_texts(), "M2").memory
            for candidate in generator.generate(item.message.text, memory):
                regenerated[item.id, candidate.style] = candidate
        metrics = {"STANDARD": [], "LOCAL": []}
        exact = 0
        for row in rows:
            candidate = regenerated[row["item_id"], row["candidate"]["style"]]
            if candidate.candidate_text != row["candidate"]["candidate_text"] or abs(candidate.score - row["candidate"]["score"]) > 1e-6:
                raise ValueError("saved generated output does not reproduce")
            item = by_id[row["item_id"]]
            gold = item.replies.standard if candidate.style == "STANDARD" else item.replies.local
            pred_counts, gold_counts = Counter(tokenize(candidate.candidate_text)), Counter(tokenize(gold))
            f1 = 2 * sum((pred_counts & gold_counts).values()) / (sum(pred_counts.values()) + sum(gold_counts.values()))
            if gold != row["reference"] or abs(f1 - row["token_f1"]) > 1e-10:
                raise ValueError("saved reference/overlap metric mismatch")
            metrics[candidate.style].append(f1)
            exact += tokenize(candidate.candidate_text) == tokenize(gold)
        if exact != comparison[version]["exact_matches"] or any(
            abs(sum(values) / len(values) - comparison[version]["token_f1"][style]) > 1e-10
            for style, values in metrics.items()
        ):
            raise ValueError("comparison metrics do not match regenerated outputs")
        results["versions"][version] = {"public_validation_loss": value, "task_validation_loss": value_task,
                                       "regenerated_outputs_matched": len(rows),
                                       "checkpoint_sha256": generator.checkpoint_hash}
        print(f"verified {version}: public/task loss and all 100 generated outputs", flush=True)
    exported = [json.loads(line) for line in (run_dir / "three_modes_v4.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row["item_id"], row["requested_mode"]) for row in exported}
    if len(exported) != 150 or keys != {(item.id, mode) for item in inputs.queries.items for mode in ("retrieval_only", "generator_only", "hybrid")}:
        raise ValueError("three-mode export coverage mismatch")
    if any(row["errors"] or row["requested_mode"] != row["effective_mode"] for row in exported):
        raise ValueError("export contains incomplete modes")
    results["system_exports"] = {"rows": len(exported), "modes": {}}
    for mode in ("retrieval_only", "generator_only", "hybrid"):
        subset = [row for row in exported if row["requested_mode"] == mode]
        selected = [row[s] for row in subset for s in ("standard", "local")]
        results["system_exports"]["modes"][mode] = {
            "fallback_replies": sum(c["source"] == "fallback" for c in selected),
            "selected_generator_replies": sum(c.get("meta", {}).get("candidate_source") == "V4" for c in selected),
            "selected_retrieval_replies": sum(c.get("meta", {}).get("candidate_source") in ("V0", "V2") for c in selected)}
    results["input_sha256"] = {"train": file_hash(DEFAULT_BANK), "validation": file_hash(DEFAULT_QUERIES)}
    (run_dir / "verification.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir), indent=2))
