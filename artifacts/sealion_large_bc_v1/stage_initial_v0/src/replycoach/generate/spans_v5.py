"""Train V5 stage 0: continue mT5's own span-corruption pre-training on Singapore English chat text.

mT5 was pre-trained by hiding random spans of web text and predicting them. This stage does the same on
NUS SMS and CoSEM text (scripts/prepare_singlish_text.py), so the model reads more Singlish before stage 1
teaches it to reply:

    text:   eh later go makan at canteen, you free anot
    input:  eh later go <extra_id_0> canteen, you <extra_id_1> anot
    target: <extra_id_0> makan at <extra_id_1> free

Stage 1 then starts from this checkpoint (``--model artifacts/<name>/best``) instead of google/mt5-*.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from replycoach.generate.train_v5 import ARTIFACTS, collate, epoch_batches, validation_loss

NOISE_DENSITY = 0.15  # mT5's settings: hide 15% of tokens in spans of about 3
MEAN_SPAN = 3


def load_texts(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows or not all(isinstance(row.get("text"), str) and row.get("source") for row in rows):
        raise ValueError(f"{path} must hold rows with 'source' and 'text'")
    return rows


def sentinel_ids(tokenizer, count: int = 100) -> list[int]:
    # Encoding the text is reliable across transformers versions; convert_tokens_to_ids is not for mT5 in v5.
    encoded = tokenizer([f"<extra_id_{index}>" for index in range(count)], add_special_tokens=False)["input_ids"]
    if any(len(ids) != 1 or ids[0] == tokenizer.unk_token_id for ids in encoded):
        raise ValueError("tokenizer has no <extra_id_N> sentinel tokens")
    return [ids[0] for ids in encoded]


def corrupt(ids: list[int], rng: random.Random, sentinels: list[int], eos_id: int) -> tuple[list[int], list[int]]:
    """Hide random spans: each span becomes one sentinel in the input, and the target lists the spans after their sentinels."""
    spans = max(1, round(len(ids) * NOISE_DENSITY / MEAN_SPAN))
    hidden = [False] * len(ids)
    for start in rng.sample(range(len(ids)), min(spans, len(ids))):
        for position in range(start, min(len(ids), start + MEAN_SPAN)):
            hidden[position] = True  # overlapping spans simply merge into a longer one
    source, target = [], []
    span_count = 0
    for position, token in enumerate(ids):
        if not hidden[position]:
            source.append(token)
            continue
        if position == 0 or not hidden[position - 1]:  # a new span starts: the same sentinel marks it in both
            source.append(sentinels[span_count])
            target.append(sentinels[span_count])
            span_count += 1
        target.append(token)
    return [*source, eos_id], [*target, eos_id]


def tokenise(rows: list[dict], tokenizer, max_tokens: int) -> list[list[int]]:
    """Token ids without the end token, cut to leave room for it."""
    ids = tokenizer([row["text"] for row in rows], add_special_tokens=False)["input_ids"]
    return [row_ids[: max_tokens - 1] for row_ids in ids if row_ids]


def train_spans(
    train_path: Path,
    validation_path: Path,
    model_name: str = "google/mt5-base",
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    warmup_steps: int = 500,
    max_tokens: int = 128,
    eval_every: int = 1000,
    log_every: int = 200,
    train_limit: int | None = None,
    output_name: str = "mt5_spans",
    output_dir: Path = ARTIFACTS,
    seed: int = 425,
) -> dict:
    if epochs < 1 or batch_size < 1 or eval_every < 1 or log_every < 1 or max_tokens < 8 or learning_rate <= 0:
        raise ValueError("epochs, batch size, intervals, token limit and learning rate must be positive and usable")
    if train_path.resolve() == validation_path.resolve():
        raise ValueError("training and validation paths must differ")
    if not output_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("output name may contain only letters, digits, underscores, and hyphens")
    best_dir = output_dir / output_name / "best"
    if best_dir.exists():
        raise ValueError(f"{best_dir} already exists; choose a new --output-name")
    torch.manual_seed(seed)

    train_rows = load_texts(train_path)[:train_limit]
    validation_rows = load_texts(validation_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentinels, eos_id, pad_id = sentinel_ids(tokenizer), tokenizer.eos_token_id, tokenizer.pad_token_id
    train_ids = tokenise(train_rows, tokenizer, max_tokens)
    # Validation spans are drawn once, so every evaluation scores the same task.
    validation_rng = random.Random(seed)
    validation_encoded = {
        source: [corrupt(ids, validation_rng, sentinels, eos_id)
                 for ids in tokenise([row for row in validation_rows if row["source"] == source], tokenizer, max_tokens)]
        for source in sorted({row["source"] for row in validation_rows})
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # full precision, as in stage 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def evaluate() -> dict[str, float]:
        return {source: validation_loss(model, encoded, 2 * batch_size, pad_id, device)
                for source, encoded in validation_encoded.items()}

    report_path = output_dir / f"{output_name}.json"
    best_dir.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "V5", "stage": "spans", "model": model_name, "train_path": str(train_path),
        "validation_path": str(validation_path), "train_chunks": len(train_ids),
        "train_chunks_by_source": {source: sum(row["source"] == source for row in train_rows)
                                   for source in sorted({row["source"] for row in train_rows})},
        "validation_chunks_by_source": {source: len(encoded) for source, encoded in validation_encoded.items()},
        "noise_density": NOISE_DENSITY, "mean_span": MEAN_SPAN, "max_tokens": max_tokens,
        "epochs_requested": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "warmup_steps": warmup_steps, "seed": seed, "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "checkpoint": str(best_dir), "history": [],
    }
    before = evaluate()  # the original mT5 on the same hidden spans
    report["history"].append({"step": 0, "validation_loss": round(sum(before.values()) / len(before), 4),
                              **{f"validation_loss_{source}": round(value, 4) for source, value in before.items()}})
    print(json.dumps(report["history"][0]), flush=True)

    best_loss, global_step = float("inf"), 0
    window_loss, window_tokens = 0.0, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # New spans every epoch, so the model never sees the same gaps twice.
        rng = random.Random(seed * 1000 + epoch)
        batches = epoch_batches(len(train_ids), batch_size, seed, epoch)
        for step_in_epoch, batch in enumerate(batches):
            model.train()
            input_ids, attention_mask, labels = collate(
                [corrupt(train_ids[index], rng, sentinels, eos_id) for index in batch], pad_id
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * min(1.0, (global_step + 1) / max(1, warmup_steps))
            optimizer.zero_grad()
            loss = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device)
            ).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is not finite at step {global_step + 1}; lower --learning-rate")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            token_count = int(labels.ne(-100).sum().item())
            window_loss += float(loss.item()) * token_count
            window_tokens += token_count
            if global_step % log_every == 0:
                print(json.dumps({"step": global_step, "epoch": epoch,
                                  "train_loss": round(window_loss / window_tokens, 4)}), flush=True)
            if global_step % eval_every != 0 and step_in_epoch != len(batches) - 1:
                continue

            losses = evaluate()
            mean_loss = sum(losses.values()) / len(losses)
            entry = {
                "step": global_step, "epoch": epoch, "train_loss": round(window_loss / window_tokens, 4),
                "validation_loss": round(mean_loss, 4),
                **{f"validation_loss_{source}": round(value, 4) for source, value in losses.items()},
                "seconds": round(time.perf_counter() - started, 1),
            }
            report["history"].append(entry)
            print(json.dumps(entry), flush=True)
            if mean_loss < best_loss:
                best_loss = mean_loss
                report["best_step"], report["best_validation_loss"] = global_step, round(mean_loss, 4)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
            if device.type == "cuda":
                report["peak_cuda_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            window_loss, window_tokens = 0.0, 0

    # A few filled-in gaps from the best checkpoint, to see what the model learned.
    model = AutoModelForSeq2SeqLM.from_pretrained(best_dir).to(device).eval()
    report["examples"] = []
    for source, encoded in validation_encoded.items():
        for source_ids, target_ids in encoded[:3]:
            with torch.no_grad():
                predicted = model.generate(input_ids=torch.tensor([source_ids], device=device), max_new_tokens=32)[0]
            report["examples"].append({
                "source": source, "input": tokenizer.decode(source_ids, skip_special_tokens=False),
                "target": tokenizer.decode(target_ids, skip_special_tokens=False),
                "predicted": tokenizer.decode(predicted, skip_special_tokens=False),
            })
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 stage 0: span corruption on Singapore English chat text.")
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--validation-text", type=Path, required=True)
    parser.add_argument("--model", default="google/mt5-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--train-limit", type=int, help="Read only this many training chunks (smoke runs).")
    parser.add_argument("--output-name", default="mt5_spans")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    report = train_spans(
        args.train_text, args.validation_text, args.model, args.epochs, args.batch_size, args.learning_rate,
        args.warmup_steps, args.max_tokens, args.eval_every, train_limit=args.train_limit,
        output_name=args.output_name, output_dir=args.output_dir,
    )
    print(f"original mT5 validation loss: {report['history'][0]['validation_loss']}; "
          f"best: {report['best_validation_loss']} (step {report['best_step']})")


if __name__ == "__main__":
    main()
