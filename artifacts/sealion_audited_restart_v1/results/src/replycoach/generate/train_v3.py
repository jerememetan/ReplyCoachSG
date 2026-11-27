"""Train V3/V4 RNNs: pilot one-batch check or DailyDialog experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from replycoach.generate import BOS, EOS, Vocab
from replycoach.generate.seq2seq import Seq2Seq
from replycoach.io import ROOT, load_items
from replycoach.schema import DialogueItem

ARTIFACTS = ROOT / "artifacts"


def style_tag(style: str) -> str:
    return "<standard>" if style == "STANDARD" else "<local>"


def pair_texts(items: list[DialogueItem]) -> list[tuple[str, str]]:
    pairs = []
    for item in items:
        context = " ".join(item.memory_texts())
        src = f"{context} {item.message.text}".strip()
        pairs.append((f"{style_tag('STANDARD')} {src}", item.replies.standard))
        pairs.append((f"{style_tag('LOCAL')} {src}", item.replies.local))
    return pairs


def collate(batch: list[tuple[list[int], list[int]]]):
    src = pad_sequence([torch.tensor(src_ids) for src_ids, _ in batch], batch_first=True, padding_value=0)
    tgt = pad_sequence([torch.tensor(tgt_ids) for _, tgt_ids in batch], batch_first=True, padding_value=0)
    src_lengths = torch.tensor([len(src_ids) for src_ids, _ in batch])
    return src, src_lengths, tgt


def run_one_batch(
    data_path: Path | None = None,
    batch_size: int = 4,
    embed_dim: int = 64,
    hidden_size: int = 128,
    use_attention: bool = False,
    seed: int = 425,
) -> dict:
    torch.manual_seed(seed)
    items = load_items(data_path)
    pairs = pair_texts(items)
    vocab = Vocab.build([text for pair in pairs for text in pair], max_size=5000)
    encoded = [(vocab.encode(src), vocab.encode(tgt)) for src, tgt in pairs]
    batch = encoded[:batch_size]
    src, src_lengths, tgt = collate(batch)
    model = Seq2Seq(len(vocab), embed_dim=embed_dim, hidden_size=hidden_size, use_attention=use_attention)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    optimizer.zero_grad()
    logits = model(src, src_lengths, tgt, teacher_forcing=True)
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ckpt_path = ARTIFACTS / ("rnn_v4_smoke.pt" if use_attention else "rnn_v3_smoke.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab.to_dict(),
            "use_attention": use_attention,
            "embed_dim": embed_dim,
            "hidden_size": hidden_size,
            "loss": float(loss.item()),
        },
        ckpt_path,
    )
    report = {
        "version": "V4" if use_attention else "V3",
        "items": len(items),
        "pairs": len(pairs),
        "batch_size": batch_size,
        "vocab_size": len(vocab),
        "loss": round(float(loss.item()), 4),
        "checkpoint": str(ckpt_path),
        "device": "cpu",
    }
    report_path = ARTIFACTS / "rnn_one_batch.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def load_public_pairs(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    """Read the output of scripts/prepare_dailydialog_pairs.py."""
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "dailydialog":
                raise ValueError(f"{path} line {line_no}: expected dailydialog source")
            context = row.get("context")
            message, response = row.get("message"), row.get("response")
            if not isinstance(context, list) or not all(isinstance(turn, str) for turn in context):
                raise ValueError(f"{path} line {line_no}: context must be a list of strings")
            if not isinstance(message, str) or not message.strip():
                raise ValueError(f"{path} line {line_no}: message must be non-empty")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"{path} line {line_no}: response must be non-empty")
            pairs.append((" ".join([*context, message]), response))
            if limit is not None and len(pairs) >= limit:
                break
    if not pairs:
        raise ValueError(f"no dialogue pairs in {path}")
    return pairs


def encode_pair(pair: tuple[str, str], vocab: Vocab, max_source: int, max_target: int):
    source, target = pair
    source_ids = vocab.encode(source)
    target_ids = vocab.encode(target)
    # Keep the latest source words, which include the incoming message.
    source_ids = [source_ids[0], *source_ids[-(max_source - 1):]] if len(source_ids) > max_source else source_ids
    target_ids = [*target_ids[: max_target - 1], target_ids[-1]] if len(target_ids) > max_target else target_ids
    return source_ids, target_ids


def epoch_loss(model, batches, criterion, device, optimizer=None) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_tokens = 0
    for src, src_lengths, tgt in batches:
        src, tgt = src.to(device), tgt.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(src, src_lengths, tgt)
            gold = tgt[:, 1:]
            loss_sum = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            token_count = int(gold.ne(0).sum().item())
            if optimizer is not None:
                (loss_sum / token_count).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss_sum.item())
        total_tokens += token_count
    return total_loss / total_tokens


@torch.no_grad()
def generate_reply(model: Seq2Seq, vocab: Vocab, source: str, max_source: int, max_target: int, device) -> str:
    model.eval()
    source_ids, _ = encode_pair((source, "placeholder"), vocab, max_source, max_target)
    src = torch.tensor([source_ids], device=device)
    encoder_outputs, hidden = model.encoder(src, torch.tensor([len(source_ids)]))
    token = torch.tensor([vocab.token_to_id[BOS]], device=device)
    generated = []
    for _ in range(max_target - 1):
        if model.use_attention:
            logits, hidden, _ = model.decoder(token, hidden, encoder_outputs)
        else:
            logits, hidden, _ = model.decoder(token, hidden)
        token = logits.argmax(dim=-1)
        token_id = int(token.item())
        if token_id == vocab.token_to_id[EOS]:
            break
        generated.append(vocab.id_to_token[token_id])
    return " ".join(generated)


def train_dailydialog(
    train_path: Path,
    validation_path: Path,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    max_vocab: int = 5000,
    max_source: int = 40,
    max_target: int = 40,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    output_name: str = "rnn_v3_dailydialog",
    seed: int = 425,
    use_attention: bool = False,
    resume_from: Path | None = None,
) -> dict:
    if epochs < 1 or batch_size < 1 or max_vocab < 4 or max_source < 3 or max_target < 3:
        raise ValueError("epochs, batch size, vocabulary, and token limits must be positive and usable")
    if (train_limit is not None and train_limit < 1) or (validation_limit is not None and validation_limit < 1):
        raise ValueError("pair limits must be positive")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if train_path.resolve() == validation_path.resolve():
        raise ValueError("training and validation paths must differ")
    if not output_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("output name may contain only letters, digits, underscores, and hyphens")
    torch.manual_seed(seed)
    train_pairs = load_public_pairs(train_path, train_limit)
    validation_pairs = load_public_pairs(validation_path, validation_limit)
    vocab = Vocab.build([text for pair in train_pairs for text in pair], max_size=max_vocab)
    train_encoded = [encode_pair(pair, vocab, max_source, max_target) for pair in train_pairs]
    validation_encoded = [encode_pair(pair, vocab, max_source, max_target) for pair in validation_pairs]
    train_batches = DataLoader(
        train_encoded, batch_size=batch_size, shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_batches = DataLoader(validation_encoded, batch_size=batch_size, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(len(vocab), use_attention=use_attention).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    checkpoint = ARTIFACTS / f"{output_name}.pt"
    report_path = ARTIFACTS / f"{output_name}.json"
    report = {
        "version": "V4" if use_attention else "V3", "source": "dailydialog", "train_path": str(train_path),
        "validation_path": str(validation_path), "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs), "epochs_requested": epochs,
        "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed,
        "max_vocab": max_vocab, "vocab_size": len(vocab),
        "max_source_tokens": max_source, "max_target_tokens": max_target,
        "device": str(device), "checkpoint": str(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": [],
    }
    best_loss = float("inf")
    start_epoch = 1
    if resume_from is not None:
        saved = torch.load(resume_from, map_location=device, weights_only=True)
        if bool(saved.get("use_attention")) != use_attention:
            raise ValueError("resume checkpoint architecture does not match --attention")
        if saved.get("vocab") != vocab.to_dict():
            raise ValueError("resume checkpoint vocabulary does not match current training data/settings")
        if saved.get("max_source_tokens") != max_source or saved.get("max_target_tokens") != max_target:
            raise ValueError("resume checkpoint token limits do not match current settings")
        model.load_state_dict(saved["model_state"])
        optimizer_restored = "optimizer_state" in saved
        if optimizer_restored:
            optimizer.load_state_dict(saved["optimizer_state"])
        completed_epoch = int(saved["epoch"])
        start_epoch = completed_epoch + 1
        if report_path.exists():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            previous_history = [entry for entry in previous.get("history", []) if entry["epoch"] <= completed_epoch]
            report["history"] = previous_history
            if previous_history:
                best_entry = min(previous_history, key=lambda entry: entry["validation_loss"])
                best_loss = float(best_entry["validation_loss"])
                report["best_epoch"] = int(best_entry["epoch"])
                report["best_validation_loss"] = best_loss
        if best_loss == float("inf"):
            best_loss = float(saved["validation_loss"])
            report["best_epoch"] = completed_epoch
            report["best_validation_loss"] = round(best_loss, 4)
        report["resumed_from"] = str(resume_from)
        report["resume_epoch"] = completed_epoch
        report["optimizer_restored"] = optimizer_restored
        if not optimizer_restored:
            report["resume_note"] = "Model weights restored; older checkpoint had no optimizer state, so Adam restarted."
        if start_epoch > epochs:
            raise ValueError(f"checkpoint already completed epoch {completed_epoch}; --epochs must be greater")

    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        training_loss = epoch_loss(model, train_batches, criterion, device, optimizer)
        validation_loss = epoch_loss(model, validation_batches, criterion, device)
        entry = {
            "epoch": epoch, "train_loss": round(training_loss, 4),
            "validation_loss": round(validation_loss, 4),
            "seconds": round(time.perf_counter() - started, 1),
        }
        report["history"].append(entry)
        if validation_loss < best_loss:
            best_loss = validation_loss
            report["best_epoch"] = epoch
            report["best_validation_loss"] = round(validation_loss, 4)
            torch.save({
                "model_state": model.state_dict(), "vocab": vocab.to_dict(),
                "use_attention": use_attention, "embed_dim": 64, "hidden_size": 128,
                "max_source_tokens": max_source, "max_target_tokens": max_target,
                "epoch": epoch, "validation_loss": validation_loss,
                "optimizer_state": optimizer.state_dict(),
            }, checkpoint)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(entry), flush=True)

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    report["examples"] = [
        {"input": source, "reference": target,
         "generated": generate_reply(model, vocab, source, max_source, max_target, device)}
        for source, target in validation_pairs[:5]
    ]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V3/V4 pilot smoke or full DailyDialog training.")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--attention", action="store_true", help="Train V4 attention-RNN instead of V3.")
    parser.add_argument("--train-pairs", type=Path)
    parser.add_argument("--validation-pairs", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-vocab", type=int, default=5000)
    parser.add_argument("--max-source-tokens", type=int, default=40)
    parser.add_argument("--max-target-tokens", type=int, default=40)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--output-name")
    parser.add_argument("--resume-from", type=Path, help="Resume to --epochs total epochs from a saved checkpoint.")
    args = parser.parse_args()
    if args.train_pairs or args.validation_pairs:
        if not args.train_pairs or not args.validation_pairs:
            parser.error("--train-pairs and --validation-pairs must be provided together")
        if args.data:
            parser.error("DailyDialog training cannot use --data")
        output_name = args.output_name or ("rnn_v4_dailydialog" if args.attention else "rnn_v3_dailydialog")
        report = train_dailydialog(
            args.train_pairs, args.validation_pairs, args.epochs,
            args.batch_size if args.batch_size is not None else 32, args.learning_rate,
            args.max_vocab, args.max_source_tokens, args.max_target_tokens,
            args.train_limit, args.validation_limit, output_name,
            use_attention=args.attention,
            resume_from=args.resume_from,
        )
        print(f"best validation loss: {report['best_validation_loss']} (epoch {report['best_epoch']})")
    else:
        report = run_one_batch(
            data_path=args.data,
            batch_size=args.batch_size if args.batch_size is not None else 4,
            use_attention=args.attention,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
