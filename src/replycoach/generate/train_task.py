"""Fine-tune existing RNNs on human-reviewed training replies only."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from replycoach.evaluation import load_evaluation_inputs
from replycoach.generate import Vocab
from replycoach.generate.rnn_provider import FORMAT, STYLE_TOKENS, file_hash, task_source
from replycoach.generate.seq2seq import Seq2Seq
from replycoach.generate.train_v3 import collate, encode_pair, epoch_loss
from replycoach.pipeline import prepare_input
from replycoach.schema import STYLES
from replycoach.tokenize import tokenize


def expand_pretrained(saved: dict, train_items: list, seed: int):
    """Retain every pretrained ID/weight; add only human TRAIN vocabulary and styles."""
    torch.manual_seed(seed)
    vocab_map = dict(saved["vocab"])
    additions = list(STYLE_TOKENS.values())
    for item in train_items:
        for text in [*item.memory_texts(), item.message.text, item.replies.standard, item.replies.local]:
            additions.extend(tokenize(text))
    for token in additions:
        if token not in vocab_map:
            vocab_map[token] = len(vocab_map)
    vocab = Vocab.from_dict(vocab_map)
    model = Seq2Seq(len(vocab), saved["embed_dim"], saved["hidden_size"], saved["use_attention"])
    state = model.state_dict()
    previous_size = len(saved["vocab"])
    expandable = {"encoder.embed.weight", "decoder.embed.weight", "decoder.out.weight", "decoder.out.bias"}
    for key, value in saved["model_state"].items():
        if key in expandable:
            state[key][:previous_size].copy_(value)
        else:
            state[key].copy_(value)
    model.load_state_dict(state)
    return model, vocab


def task_pairs(items, vocab, max_source, max_target):
    pairs = []
    for item in items:
        memory = prepare_input(item.message.text, item.memory_texts(), "M2").memory
        for style, reply in zip(STYLES, (item.replies.standard, item.replies.local)):
            src = task_source(vocab, style, item.message.text, memory, max_source)
            _, tgt = encode_pair(("", reply), vocab, max_source, max_target)
            pairs.append((src, tgt))
    return pairs


def train_task(pretrained: Path, train_path: Path, validation_path: Path, output_dir: Path,
               *, epochs: int = 60, patience: int = 10, seed: int = 425,
               batch_size: int = 32, learning_rate: float = 1e-3,
               max_source: int = 64, max_target: int = 40) -> dict:
    if min(epochs, patience, batch_size) < 1 or learning_rate <= 0 or max_source < 4 or max_target < 3:
        raise ValueError("invalid task training configuration")
    inputs = load_evaluation_inputs(validation_path, train_path)
    saved = torch.load(pretrained, map_location="cpu", weights_only=True)
    if "epoch" not in saved or "validation_loss" not in saved or saved.get("format") == FORMAT:
        raise ValueError("expected a public-pretrained checkpoint, not smoke/task weights")
    model, vocab = expand_pretrained(saved, inputs.bank.items, seed)
    train_encoded = task_pairs(inputs.bank.items, vocab, max_source, max_target)
    val_encoded = task_pairs(inputs.queries.items, vocab, max_source, max_target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batches = DataLoader(train_encoded, batch_size=batch_size, shuffle=True, collate_fn=collate,
                         generator=torch.Generator().manual_seed(seed))
    val_batches = DataLoader(val_encoded, batch_size=batch_size, collate_fn=collate)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    version = "V4" if saved["use_attention"] else "V3"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{version.lower()}_task.pt"
    report_path = output_dir / f"{version.lower()}_task.json"
    if checkpoint.exists() or report_path.exists():
        raise FileExistsError("task outputs exist; use a fresh run directory")
    provenance = {"pretrained_sha256": file_hash(pretrained),
                  "train_sha256": file_hash(train_path), "validation_sha256": file_hash(validation_path),
                  "train_ids": [i.id for i in inputs.bank.items],
                  "validation_ids": [i.id for i in inputs.queries.items]}
    report = {"version": version, "training_stage": "human_reviewed_task_finetuning",
              "pretrained": str(pretrained), "checkpoint": str(checkpoint),
              "provenance": provenance, "seed": seed, "device": str(device),
              "train_items": len(inputs.bank.items), "validation_items": len(inputs.queries.items),
              "train_pairs": len(train_encoded), "validation_pairs": len(val_encoded),
              "epochs_requested": epochs, "patience": patience, "batch_size": batch_size,
              "learning_rate": learning_rate, "vocab_size": len(vocab), "memory_mode": "M2",
              "max_source_tokens": max_source, "max_target_tokens": max_target,
              "selection": "minimum validation token cross entropy; no test access",
              "initial_validation_loss": epoch_loss(model, val_batches, criterion, device), "history": []}
    best, stale = float("inf"), 0
    started_all = time.perf_counter()
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = epoch_loss(model, batches, criterion, device, optimizer)
        val_loss = epoch_loss(model, val_batches, criterion, device)
        entry = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                 "seconds": round(time.perf_counter() - started, 3)}
        report["history"].append(entry)
        if val_loss < best:
            best, stale = val_loss, 0
            report.update(best_epoch=epoch, best_validation_loss=val_loss)
            torch.save({"format": FORMAT, "task_trained": True, "model_state": model.state_dict(),
                        "vocab": vocab.to_dict(), "use_attention": saved["use_attention"],
                        "embed_dim": saved["embed_dim"], "hidden_size": saved["hidden_size"],
                        "max_source_tokens": max_source, "max_target_tokens": max_target,
                        "epoch": epoch, "validation_loss": val_loss, "provenance": provenance}, checkpoint)
        else:
            stale += 1
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"stage": f"{version}_task", **entry}), flush=True)
        if stale >= patience:
            break
    report.update(epochs_completed=len(report["history"]),
                  total_seconds=round(time.perf_counter() - started_all, 3),
                  checkpoint_sha256=file_hash(checkpoint))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
