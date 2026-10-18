"""Train V5 stage 2: fine-tune the stage 1 mT5 on team-authored development items with LoRA.

Each item gives two pairs in the ``mt5_input`` format stage 1 used:
``STANDARD context: ... message: ...`` -> ``replies.standard`` and ``LOCAL ...`` -> ``replies.local``.
One adapter learns both styles and the tag chooses between them. Only development items are read:
data/dev/train.jsonl trains, data/dev/validation.jsonl picks the epoch. The test set never enters.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import AutoPeftModelForSeq2SeqLM, LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from replycoach.generate.train_v5 import (
    ARTIFACTS, collate, encode_rows, epoch_batches, generate_replies, load_pairs, validation_loss,
)
from replycoach.io import DATA_DIR, load_items

STYLES = ("STANDARD", "LOCAL")
DEV_TRAIN = DATA_DIR / "dev" / "train.jsonl"
DEV_VALIDATION = DATA_DIR / "dev" / "validation.jsonl"


def item_rows(path: Path) -> list[dict]:
    """Two rows per development item, one per style, sharing the item's context and message."""
    items = load_items(path)
    not_dev = [item.id for item in items if item.split != "dev"]
    if not_dev:
        raise ValueError(f"{path}: only development items may be used, found {', '.join(not_dev[:3])}")
    if not items:
        raise ValueError(f"no development items in {path}")
    return [
        {"item_id": item.id, "style": style, "context": item.memory_texts(), "message": item.message.text,
         "response": response}
        for item in items
        for style, response in (("STANDARD", item.replies.standard), ("LOCAL", item.replies.local))
    ]


def shorter_contexts(row: dict) -> list[dict]:
    """Copies of a row with less context: the last turn only, then none.

    The app's M0/M1/M2 memory modes pass different amounts of context, and a model that also sees the bare
    message has to answer the message instead of echoing the context. The reply stays the team's own text.
    """
    return [{**row, "context": context} for context in (row["context"][-1:], [])
            if len(context) < len(row["context"])]


def load_finetuned(checkpoint: Path, device="cpu"):
    """A LoRA checkpoint holds only the adapter and names its base model; a full fine-tune holds everything."""
    if (checkpoint / "adapter_config.json").exists():
        return AutoPeftModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    return AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)


def finetune(
    base_model: str,
    train_path: Path = DEV_TRAIN,
    validation_path: Path = DEV_VALIDATION,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_targets: tuple[str, ...] = ("q", "v"),
    epochs: int = 30,
    patience: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    max_source: int = 128,
    max_target: int = 48,
    output_name: str = "mt5_v5_lora",
    output_dir: Path = ARTIFACTS,
    seed: int = 425,
    context_variants: bool = False,
    rehearsal_paths: list[Path] = (),
    rehearsal_count: int = 400,
) -> dict:
    """Fine-tune ``base_model`` (a stage 1 checkpoint). ``lora_rank=0`` trains every weight instead of an adapter.

    Augmentation touches training only, so every run is scored on the same unchanged validation pairs:
    ``context_variants`` adds shorter-context copies of each item, and ``rehearsal_paths`` mixes in
    ``rehearsal_count`` stage 1 public pairs so the model keeps its general reply ability.
    """
    if epochs < 1 or patience < 1 or batch_size < 1 or lora_rank < 0 or learning_rate <= 0 or rehearsal_count < 0:
        raise ValueError("epochs, patience and batch size must be positive, rank and rehearsal count "
                         "non-negative, learning rate positive")
    if not output_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("output name may contain only letters, digits, underscores, and hyphens")
    run_dir = output_dir / output_name
    best_dir = run_dir / "best"
    # Runs take minutes, so there is no resume: a reused name would mix two runs' files.
    if best_dir.exists():
        raise ValueError(f"{best_dir} already exists; choose a new --output-name")
    torch.manual_seed(seed)

    train_rows = item_rows(train_path)
    validation_rows = item_rows(validation_path)
    overlap = {row["item_id"] for row in train_rows} & {row["item_id"] for row in validation_rows}
    if overlap:
        raise ValueError(f"items in both training and validation: {', '.join(sorted(overlap)[:3])}")
    item_pairs = len(train_rows)
    variants = [variant for row in train_rows for variant in shorter_contexts(row)] if context_variants else []
    public = [row for path in rehearsal_paths for row in load_pairs(path)]
    rehearsal = random.Random(seed).sample(public, min(rehearsal_count, len(public)))
    train_rows = [*train_rows, *variants, *rehearsal]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pad_id = tokenizer.pad_token_id
    train_encoded = encode_rows(train_rows, tokenizer, max_source, max_target)
    # Validated per style, so the report shows if one style lags behind the other.
    validation_encoded = {
        style: encode_rows([row for row in validation_rows if row["style"] == style], tokenizer, max_source, max_target)
        for style in STYLES
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Full precision, as in stage 1: mT5 overflows under fp16.
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def evaluate() -> dict[str, float]:
        return {style: validation_loss(model, encoded, 2 * batch_size, pad_id, device)
                for style, encoded in validation_encoded.items()}

    # The stage 1 model alone on the same items: the number fine-tuning has to beat.
    before = evaluate()

    if lora_rank > 0:
        # Freeze every stage 1 weight and add trainable rank-r matrices beside the chosen attention projections.
        # B starts at zero, so the wrapped model starts out identical to the stage 1 model.
        config = LoraConfig(
            task_type="SEQ_2_SEQ_LM", r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(lora_targets),
        )
        model = get_peft_model(model, config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    report_path = output_dir / f"{output_name}.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "V5", "stage": "finetune", "method": "lora" if lora_rank > 0 else "full",
        "base_model": str(base_model), "train_path": str(train_path), "validation_path": str(validation_path),
        "train_items": item_pairs // 2, "train_pairs": len(train_encoded),
        "train_pairs_by_kind": {"items": item_pairs, "context_variants": len(variants), "rehearsal": len(rehearsal)},
        "rehearsal_paths": [str(path) for path in rehearsal_paths],
        "validation_items": len(validation_rows) // 2, "seed": seed,
        "lora": {"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout, "targets": list(lora_targets)}
                if lora_rank > 0 else None,
        "epochs_requested": epochs, "patience": patience, "batch_size": batch_size, "learning_rate": learning_rate,
        "max_source_tokens": max_source, "max_target_tokens": max_target,
        "device": str(device), "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "checkpoint": str(best_dir),
        "history": [{
            "epoch": 0, "validation_loss": round(sum(before.values()) / len(before), 4),
            **{f"validation_loss_{style.lower()}": round(value, 4) for style, value in before.items()},
        }],
    }
    print(json.dumps(report["history"][0]), flush=True)

    best_loss, stale = float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_tokens = 0.0, 0
        for batch in epoch_batches(len(train_encoded), batch_size, seed, epoch):
            input_ids, attention_mask, labels = collate([train_encoded[index] for index in batch], pad_id)
            optimizer.zero_grad()
            loss = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device)
            ).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is not finite in epoch {epoch}; lower --learning-rate")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            token_count = int(labels.ne(-100).sum().item())
            epoch_loss += float(loss.item()) * token_count
            epoch_tokens += token_count

        losses = evaluate()
        # Unweighted mean, so neither style decides the checkpoint alone.
        mean_loss = sum(losses.values()) / len(losses)
        entry = {
            "epoch": epoch, "train_loss": round(epoch_loss / epoch_tokens, 4), "validation_loss": round(mean_loss, 4),
            **{f"validation_loss_{style.lower()}": round(value, 4) for style, value in losses.items()},
            "seconds": round(time.perf_counter() - started, 1),
        }
        report["history"].append(entry)
        print(json.dumps(entry), flush=True)
        if mean_loss < best_loss:
            best_loss, stale = mean_loss, 0
            report["best_epoch"], report["best_validation_loss"] = epoch, round(mean_loss, 4)
            model.save_pretrained(best_dir)  # a LoRA model saves only its adapter
            tokenizer.save_pretrained(best_dir)
        else:
            stale += 1
        if device.type == "cuda":
            report["peak_cuda_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if stale >= patience:
            # Validation has stopped improving; more epochs would only memorise the training items.
            report["stopped_early"] = True
            break

    # Replies from the best epoch for every validation item, to read and to compare with V4.
    model = load_finetuned(best_dir, device)
    references = {row["item_id"]: row for row in validation_rows if row["style"] == "STANDARD"}
    local_references = {row["item_id"]: row["response"] for row in validation_rows if row["style"] == "LOCAL"}
    rows = list(references.values())
    generated = {
        style: generate_replies(model, tokenizer, rows, style, max_source, max_target, device) for style in STYLES
    }
    report["validation_outputs"] = [
        {
            "item_id": row["item_id"], "context": row["context"], "message": row["message"],
            "reference_standard": row["response"], "generated_standard": generated["STANDARD"][index],
            "reference_local": local_references[row["item_id"]], "generated_local": generated["LOCAL"][index],
        }
        for index, row in enumerate(rows)
    ]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 stage 2: LoRA fine-tune of a stage 1 mT5 on development items.")
    parser.add_argument("--base-model", required=True, help="Stage 1 checkpoint, e.g. artifacts/mt5_v5_pretrain/best")
    parser.add_argument("--train-items", type=Path, default=DEV_TRAIN)
    parser.add_argument("--validation-items", type=Path, default=DEV_VALIDATION)
    parser.add_argument("--lora-rank", type=int, default=16, help="0 trains every weight instead of an adapter.")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-targets", nargs="+", default=["q", "v"], help="mT5 layers: q k v o wi_0 wi_1 wo")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5, help="Stop after this many epochs without improvement.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-source-tokens", type=int, default=128)
    parser.add_argument("--max-target-tokens", type=int, default=48)
    parser.add_argument("--output-name", default="mt5_v5_lora")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--context-variants", action="store_true", help="Also train each item with less context.")
    parser.add_argument("--rehearsal-pairs", type=Path, nargs="+", default=[],
                        help="Stage 1 pair files to mix into training, e.g. data/public/cosem_train_pairs.jsonl")
    parser.add_argument("--rehearsal-count", type=int, default=400, help="Public pairs to mix in from those files.")
    args = parser.parse_args()
    report = finetune(
        args.base_model, args.train_items, args.validation_items, args.lora_rank, args.lora_alpha,
        args.lora_dropout, tuple(args.lora_targets), args.epochs, args.patience, args.batch_size,
        args.learning_rate, args.max_source_tokens, args.max_target_tokens, args.output_name, args.output_dir,
        context_variants=args.context_variants, rehearsal_paths=args.rehearsal_pairs,
        rehearsal_count=args.rehearsal_count,
    )
    print(f"stage 1 validation loss: {report['history'][0]['validation_loss']}; "
          f"best: {report['best_validation_loss']} (epoch {report['best_epoch']})")


if __name__ == "__main__":
    main()
