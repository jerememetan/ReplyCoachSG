"""Train V5 stage 1: adapt mT5-small to dialogue on public pairs before the dual-style fine-tune.

DailyDialog replies are tagged STANDARD and CoSEM replies LOCAL, in the same
``mt5_input`` format the fine-tune uses, so the style prefix already means
something before the model sees team-authored items. Only public pairs are read
here; no ReplyCoach development or test item enters this stage.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from replycoach.generate.style_control import mt5_input
from replycoach.io import ROOT

ARTIFACTS = ROOT / "artifacts"
STYLE_BY_SOURCE = {"dailydialog": "STANDARD", "cosem": "LOCAL"}
STATE_FILE = "training_state.pt"


def detokenise(text: str) -> str:
    """DailyDialog is pre-tokenised ("Hello , how are you ?"); mT5 should learn normal spacing."""
    text = re.sub(r"\s+([,.?!;:])", r"\1", text)
    text = re.sub(r"\s+(['’])\s*(s|m|re|ve|ll|d|t)\b", r"\1\2", text)
    return " ".join(text.split())


def load_pairs(path: Path, limit: int | None = None) -> list[dict]:
    """Read the output of scripts/prepare_dailydialog_pairs.py or scripts/prepare_cosem_pairs.py."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source")
            if source not in STYLE_BY_SOURCE:
                raise ValueError(f"{path} line {line_no}: source must be one of {sorted(STYLE_BY_SOURCE)}")
            context, message, response = row.get("context"), row.get("message"), row.get("response")
            if not isinstance(context, list) or not all(isinstance(turn, str) for turn in context):
                raise ValueError(f"{path} line {line_no}: context must be a list of strings")
            if not isinstance(message, str) or not message.strip():
                raise ValueError(f"{path} line {line_no}: message must be non-empty")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"{path} line {line_no}: response must be non-empty")
            clean = detokenise if source == "dailydialog" else lambda text: " ".join(text.split())
            rows.append({
                "source": source, "context": [clean(turn) for turn in context],
                "message": clean(message), "response": clean(response),
            })
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no dialogue pairs in {path}")
    return rows


def encode_rows(rows: list[dict], tokenizer, max_source: int, max_target: int, style: str | None = None):
    """Tokenise rows into (source ids, target ids).

    The tag is ``style`` if given, else the row's own ``style`` (team items), else the one implied by its source.
    """
    styles = [style or row.get("style") or STYLE_BY_SOURCE[row["source"]] for row in rows]
    sources = tokenizer([mt5_input(tag, row["message"], row["context"]) for tag, row in zip(styles, rows)])["input_ids"]
    targets = tokenizer([row["response"] for row in rows], max_length=max_target, truncation=True)["input_ids"]
    for index, row in enumerate(rows):
        context = list(row["context"])
        # The message sits at the end of the input, so drop the oldest turns before cutting the tail.
        while len(sources[index]) > max_source and context:
            context = context[1:]
            sources[index] = tokenizer(mt5_input(styles[index], row["message"], context))["input_ids"]
        if len(sources[index]) > max_source:
            sources[index] = [*sources[index][: max_source - 1], tokenizer.eos_token_id]
    return list(zip(sources, targets))


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int):
    input_ids = pad_sequence([torch.tensor(src) for src, _ in batch], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([torch.tensor(tgt) for _, tgt in batch], batch_first=True, padding_value=-100)
    return input_ids, input_ids.ne(pad_id), labels


def epoch_batches(count: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """Shuffle deterministically per epoch so a resumed run continues the same order."""
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    return [order[start : start + batch_size] for start in range(0, count, batch_size)]


@torch.no_grad()
def validation_loss(model, encoded: list[tuple[list[int], list[int]]], batch_size: int, pad_id: int, device) -> float:
    """Token-averaged cross entropy under teacher forcing."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(encoded), batch_size):
        input_ids, attention_mask, labels = collate(encoded[start : start + batch_size], pad_id)
        loss = model(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device)
        ).loss
        token_count = int(labels.ne(-100).sum().item())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
    return total_loss / total_tokens


@torch.no_grad()
def generate_replies(model, tokenizer, rows: list[dict], style: str, max_source: int, max_target: int, device):
    model.eval()
    encoded = encode_rows(rows, tokenizer, max_source, max_target, style=style)
    input_ids, attention_mask, _ = collate(encoded, tokenizer.pad_token_id)
    output = model.generate(
        input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
        max_new_tokens=max_target, num_beams=4, no_repeat_ngram_size=3,
    )
    return tokenizer.batch_decode(output, skip_special_tokens=True)


def save_state(path: Path, payload: dict) -> None:
    # Kaggle can stop a session mid-write; never leave a half-written resume file.
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_pretrain(
    train_paths: list[Path],
    validation_paths: list[Path],
    model_name: str = "google/mt5-small",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    warmup_steps: int = 1000,
    max_source: int = 128,
    max_target: int = 48,
    eval_every: int = 2000,
    log_every: int = 200,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    output_name: str = "mt5_v5_pretrain",
    output_dir: Path = ARTIFACTS,
    seed: int = 425,
    resume_from: Path | None = None,
) -> dict:
    if epochs < 1 or batch_size < 1 or eval_every < 1 or log_every < 1 or max_source < 8 or max_target < 3:
        raise ValueError("epochs, batch size, intervals, and token limits must be positive and usable")
    if (train_limit is not None and train_limit < 1) or (validation_limit is not None and validation_limit < 1):
        raise ValueError("pair limits must be positive")
    if learning_rate <= 0 or warmup_steps < 0:
        raise ValueError("learning rate must be positive and warmup steps non-negative")
    if {path.resolve() for path in train_paths} & {path.resolve() for path in validation_paths}:
        raise ValueError("training and validation paths must differ")
    if not output_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("output name may contain only letters, digits, underscores, and hyphens")
    torch.manual_seed(seed)
    # The limits apply per file so a smoke run still sees every source.
    train_rows = [row for path in train_paths for row in load_pairs(path, train_limit)]
    validation_rows = [row for path in validation_paths for row in load_pairs(path, validation_limit)]
    train_sources = sorted({row["source"] for row in train_rows})
    missing = [source for source in train_sources if not any(row["source"] == source for row in validation_rows)]
    if missing:
        raise ValueError(f"no validation pairs for training source(s): {', '.join(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = tokenizer.pad_token_id
    train_encoded = encode_rows(train_rows, tokenizer, max_source, max_target)
    validation_encoded = {
        source: encode_rows([row for row in validation_rows if row["source"] == source], tokenizer, max_source, max_target)
        for source in train_sources
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Full precision on purpose: mT5 activations overflow under fp16 autocast and the loss turns to NaN.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    run_dir = output_dir / output_name
    best_dir = run_dir / "best"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{output_name}.json"
    steps_per_epoch = (len(train_encoded) + batch_size - 1) // batch_size
    settings = {
        "model": model_name, "batch_size": batch_size, "seed": seed, "train_pairs": len(train_encoded),
        "max_source_tokens": max_source, "max_target_tokens": max_target,
    }
    report = {
        "version": "V5", "stage": "pretrain", **settings,
        "train_paths": [str(path) for path in train_paths],
        "validation_paths": [str(path) for path in validation_paths],
        "train_pairs_by_source": {
            source: sum(row["source"] == source for row in train_rows) for source in train_sources
        },
        "validation_pairs_by_source": {source: len(rows) for source, rows in validation_encoded.items()},
        "style_by_source": {source: STYLE_BY_SOURCE[source] for source in train_sources},
        "epochs_requested": epochs, "steps_per_epoch": steps_per_epoch,
        "learning_rate": learning_rate, "warmup_steps": warmup_steps, "eval_every": eval_every,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(best_dir), "history": [],
    }
    global_step = 0
    start_epoch, start_step = 1, 0
    best_loss = float("inf")
    trained_seconds = 0.0
    if resume_from is not None:
        saved = torch.load(resume_from, map_location=device, weights_only=True)
        if saved["settings"] != settings:
            raise ValueError(f"resume checkpoint settings {saved['settings']} do not match this run {settings}")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        global_step, start_epoch, start_step = saved["global_step"], saved["epoch"], saved["step_in_epoch"]
        best_loss, trained_seconds = saved["best_loss"], saved["trained_seconds"]
        report["history"] = saved["history"]
        report["resumed_from"] = str(resume_from)
        report["resume_step"] = global_step
        # A new Kaggle session mounts the old run read-only; carry its best model into this run.
        previous_best = resume_from.parent / "best"
        if previous_best.is_dir() and previous_best.resolve() != best_dir.resolve():
            shutil.copytree(previous_best, best_dir, dirs_exist_ok=True)
        if not best_dir.is_dir():
            best_loss = float("inf")  # no best model came with the state file; the next validation saves one
        if global_step >= epochs * steps_per_epoch:
            raise ValueError(f"checkpoint already completed step {global_step}; --epochs must be greater")
    if report["history"]:
        best_entry = min(report["history"], key=lambda entry: entry["validation_loss"])
        report["best_step"] = best_entry["step"]
        report["best_validation_loss"] = best_entry["validation_loss"]

    window_loss, window_tokens = 0.0, 0
    window_started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        batches = epoch_batches(len(train_encoded), batch_size, seed, epoch)
        for step_in_epoch in range(start_step, len(batches)):
            model.train()
            input_ids, attention_mask, labels = collate([train_encoded[index] for index in batches[step_in_epoch]], pad_id)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * min(1.0, (global_step + 1) / max(1, warmup_steps))
            optimizer.zero_grad()
            loss = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device)
            ).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is not finite at step {global_step + 1}; lower --learning-rate and resume")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            token_count = int(labels.ne(-100).sum().item())
            window_loss += float(loss.item()) * token_count
            window_tokens += token_count
            if global_step % log_every == 0:
                print(json.dumps({"step": global_step, "epoch": epoch, "train_loss": round(window_loss / window_tokens, 4)}), flush=True)
            if global_step % eval_every != 0 and step_in_epoch != len(batches) - 1:
                continue

            losses = {
                source: validation_loss(model, encoded, 2 * batch_size, pad_id, device)
                for source, encoded in validation_encoded.items()
            }
            # Unweighted mean, so the larger validation file does not decide the checkpoint.
            mean_loss = sum(losses.values()) / len(losses)
            trained_seconds += time.perf_counter() - window_started
            entry = {
                "step": global_step, "epoch": epoch, "train_loss": round(window_loss / window_tokens, 4),
                "validation_loss": round(mean_loss, 4),
                **{f"validation_loss_{source}": round(value, 4) for source, value in losses.items()},
                "seconds": round(trained_seconds, 1),
            }
            report["history"].append(entry)
            if mean_loss < best_loss:
                best_loss = mean_loss
                report["best_step"] = global_step
                report["best_validation_loss"] = round(mean_loss, 4)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
            save_state(run_dir / STATE_FILE, {
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "settings": settings, "global_step": global_step, "epoch": epoch,
                "step_in_epoch": step_in_epoch + 1, "best_loss": best_loss,
                "trained_seconds": trained_seconds, "history": report["history"],
            })
            if device.type == "cuda":
                report["peak_cuda_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(entry), flush=True)
            window_loss, window_tokens = 0.0, 0
            window_started = time.perf_counter()
        start_step = 0

    model = AutoModelForSeq2SeqLM.from_pretrained(best_dir).to(device)
    report["examples"] = []
    for source in train_sources:
        rows = [row for row in validation_rows if row["source"] == source][:3]
        replies = {
            style: generate_replies(model, tokenizer, rows, style, max_source, max_target, device)
            for style in ("STANDARD", "LOCAL")
        }
        for index, row in enumerate(rows):
            report["examples"].append({
                "source": source, "context": row["context"], "message": row["message"],
                "reference": row["response"],
                "generated_standard": replies["STANDARD"][index], "generated_local": replies["LOCAL"][index],
            })
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 stage 1: mT5 dialogue training on public pairs.")
    parser.add_argument("--train-pairs", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-pairs", type=Path, nargs="+", required=True)
    parser.add_argument("--model", default="google/mt5-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-source-tokens", type=int, default=128)
    parser.add_argument("--max-target-tokens", type=int, default=48)
    parser.add_argument("--eval-every", type=int, default=2000, help="Validate and checkpoint every N steps.")
    parser.add_argument("--train-limit", type=int, help="Pairs to read from each training file.")
    parser.add_argument("--validation-limit", type=int, help="Pairs to read from each validation file.")
    parser.add_argument("--output-name", default="mt5_v5_pretrain")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--resume-from", type=Path, help=f"Resume to --epochs total epochs from a saved {STATE_FILE}.")
    args = parser.parse_args()
    report = train_pretrain(
        args.train_pairs, args.validation_pairs, args.model, args.epochs, args.batch_size,
        args.learning_rate, args.warmup_steps, args.max_source_tokens, args.max_target_tokens,
        args.eval_every, train_limit=args.train_limit, validation_limit=args.validation_limit,
        output_name=args.output_name, output_dir=args.output_dir, resume_from=args.resume_from,
    )
    print(f"best validation loss: {report['best_validation_loss']} (step {report['best_step']})")


if __name__ == "__main__":
    main()
